"""The LLM review on top of the BO loop.

The LLM does NOT replace the GP. It gets everything the GP produced and acts as a second
opinion with domain knowledge:

    accept the main-GP suggestion?   -> yes / no, with a reason
    if no                            -> a concrete alternative recipe, with a reason

Inputs (the "packet", build_packet): the objective and its goal, the parameters with roles and
bounds, the best runs (any context and at the fixed context), the GP suggestion with its
predicted value +/- std and acquisition value, the auxiliary predictions at that recipe, the
next-best alternatives with the same predictions, the ARD lengthscales, the observed runs, and
the acquisition figure. The system prompt is built from the live spec (system_prompt): experiment
description, objective, and the spec's `llm_context` physics notes. Output: strict JSON, parsed
and also saved as Markdown.

Two ways to get the review
    1. Through a coding agent. When the package is driven from Claude Code or Codex, the agent
       itself is the reviewer: `aln-bo step` writes results/<...>/llm_prompt.md (system prompt +
       packet), the agent reads it and answers. No API key is needed, only the agent's own
       subscription.
    2. Through an API (`--llm`). review() sends the same system prompt and packet to a model
       provider: `--llm-provider anthropic` (anthropic SDK, ANTHROPIC_API_KEY; default model
       claude-opus-5-5 with adaptive thinking) or `--llm-provider openai` (openai SDK, OPENAI_API_KEY).
       Any other provider or a local model plugs in through review(call=...), a function that takes
       (system, text, png_bytes, model) and returns (reply, model_name, usage). API use is billed by
       the provider, separately from a chat subscription. Without credentials the step still writes
       llm_prompt.md. Only the Anthropic path has been exercised in this project.
"""
from __future__ import annotations
import base64
import json
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from . import config as cfg
from .config import PARAMS, PARAM_NAMES, FREE_NAMES, FIXED, AUX, PHYSICAL

DEFAULT_MODEL = "claude-opus-5-5"

PROMPT_HEAD = """You are a thin-film growth scientist advising an experimentalist who is running
Bayesian optimisation (BO) of a growth recipe. The BO engine has already fitted Gaussian processes (GPs)
and proposed the next experiment. Your job is to review that proposal, not to replace the engine.

Experiment: {display_name}. {description}
Objective: {objective} ({goal}).

How to read the inputs
- pred_* and *_std are GP posterior mean and standard deviation at a recipe. A std comparable to the
  spread of the data means the GP is extrapolating; treat such predictions as weak.
- The acquisition value (EI, Expected Improvement, unless stated otherwise) is the engine's score: large
  when the GP expects a better objective than the best run so far OR is very uncertain there. The main
  suggestion is its maximum.
- Auxiliary GPs predict secondary properties; they are descriptive, the user did not ask to optimise them.
- ARD lengthscales are in normalised [0,1] units; a SMALL value means the target changes quickly along
  that parameter (influential), a value near the upper bound means the parameter looks irrelevant.
- Context parameters are held fixed by the engine (they are part of the standard recipe); do not
  propose changing them unless you say explicitly why.

Physics you may bring in (use judgement, these are tendencies not laws)
{llm_context}

Decision rules
- ACCEPT the engine's suggestion when it is plausible, safe for the tool, and either exploits a
  credible trend or explores a region the data genuinely do not cover.
- REJECT it when it extrapolates into a corner with no supporting data for a dubious physical reason,
  when the auxiliary predictions indicate an obviously bad outcome, when it duplicates an existing
  run within instrument resolution, or when a listed alternative is clearly better justified.
- If you reject, propose ONE concrete alternative recipe. Prefer one of the listed alternatives; you
  may adjust values, but stay inside the bounds and on the instrument grid (steps given).
- Be concise and quantitative. Cite the numbers you relied on.

Respond with a single JSON object and nothing else:
{{
  "decision": "accept" | "reject",
  "recommended_recipe": {{<free parameter name>: <value>, ...}},
  "reason": "<3-6 sentences>",
  "risks": ["<short bullet>", ...],
  "confidence": "low" | "medium" | "high"
}}"""


def system_prompt(target=None) -> str:
    """The system prompt for the live experiment spec (physics notes come from spec['llm_context'])."""
    spec = cfg.current_spec()
    t = cfg.target(target) if target else cfg.MAIN
    return PROMPT_HEAD.format(display_name=spec.get("display_name", spec.get("name", "")),
                              description=spec.get("description", ""),
                              objective=f"{t.name} = {t.column}" + (f" [{t.unit}]" if t.unit else ""),
                              goal="smaller is better" if t.minimize else "larger is better",
                              llm_context=spec.get("llm_context", "- (no physics notes given for this experiment)"))


SYSTEM_PROMPT = system_prompt()        # for the default experiment; use system_prompt() after configure()


# ------------------------------------------------------------------ packet
def md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    if max_rows is not None:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.4g}"
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _rows(df: pd.DataFrame, cols, n=None) -> list[dict]:
    d = df[cols].round(4)
    if n is not None:
        d = d.head(n)
    return json.loads(d.to_json(orient="records"))


def build_packet(main, aux, suggestion: pd.DataFrame, alternatives: pd.DataFrame, dataset,
                 aux_at_suggestion: pd.DataFrame, aux_at_alternatives: pd.DataFrame) -> dict:
    t = main.target.name
    best = main.best_so_far()
    params = [{"name": p.name, "column": p.column, "unit": p.unit, "low": p.low, "high": p.high,
               "step": p.step, "role": "free" if p.is_free else f"context (fixed at {p.fixed})"} for p in PARAMS]
    runs_cols = [c for c in ["run"] + PARAM_NAMES + list(PHYSICAL) + [t] + [a.name for a in AUX] if c in dataset.df.columns]
    observed = dataset.df[runs_cols].sort_values(t).reset_index(drop=True)
    sug = pd.concat([suggestion.reset_index(drop=True), aux_at_suggestion.reset_index(drop=True)], axis=1)
    alt = pd.concat([alternatives.reset_index(drop=True), aux_at_alternatives.reset_index(drop=True)], axis=1)
    bic = main.best_in_context()
    return {
        "objective": {"name": t, "column": main.target.column, "goal": "minimize" if main.target.minimize else "maximize",
                      "n_runs_used": int(len(main.X)), "best_observed_any_context": float(main.Y.min() if main.target.minimize else main.Y.max()),
                      "ei_incumbent_in_context": float(main.incumbent()),
                      "gp_noise_std": main.noise(), "backend": main.backend_name,
                      "irrelevant_parameters_pinned": main.irrelevant()},
        "parameters": params,
        "best_run_any_context": json.loads(best.to_json()),
        "best_run_at_fixed_context": None if bic is None else json.loads(bic.to_json()),
        "main_gp_suggestion": json.loads(sug.round(4).to_json(orient="records")),
        "alternatives_by_EI": json.loads(alt.round(4).to_json(orient="records")),
        "main_gp_lengthscales": main.importance(),
        "auxiliary_models": {name: {"n_runs": aux.n_rows[name], "lengthscales": aux.importance().get(name),
                                    "observed_range": aux.best_observed().get(name)} for name in aux.n_rows},
        "observed_runs_sorted_by_objective": _rows(observed, runs_cols),
    }


def packet_to_markdown(packet: dict) -> str:
    o = packet["objective"]
    noise = o["gp_noise_std"]
    lines = [f"# BO review packet - objective {o['name']} ({o['goal']})", "",
             f"Runs used: {o['n_runs_used']}  |  best observed (any context): {o['best_observed_any_context']:.4f}  |  "
             f"EI incumbent at the fixed context: {o['ei_incumbent_in_context']:.4f}  |  "
             f"GP noise std: {noise if noise is None else round(noise, 4)}  |  backend: {o['backend']}",
             f"Parameters the main GP finds irrelevant (pinned to the best in-context run): {o['irrelevant_parameters_pinned']}", "",
             "## Parameters", "| name | column | role | bounds | step |", "|---|---|---|---|---|"]
    for p in packet["parameters"]:
        lines.append(f"| {p['name']} | {p['column']} | {p['role']} | [{p['low']}, {p['high']}] {p['unit']} | {p['step']} |")
    lines += ["", "## Best run at the fixed context (the one to beat)", "```json",
              json.dumps(packet["best_run_at_fixed_context"], indent=1), "```",
              "", "## Best run over any context (for reference; different growth time)", "```json",
              json.dumps(packet["best_run_any_context"], indent=1), "```",
              "", "## Main GP suggestion (EI maximum) with auxiliary predictions",
              md_table(pd.DataFrame(packet["main_gp_suggestion"])), "",
              "## Next-best alternatives by EI (distinct recipes)",
              md_table(pd.DataFrame(packet["alternatives_by_EI"])), "",
              "## ARD lengthscales (small = influential)",
              f"main: {packet['main_gp_lengthscales']}"]
    for name, info in packet["auxiliary_models"].items():
        lines.append(f"{name}: n={info['n_runs']}, lengthscales={info['lengthscales']}, observed={info['observed_range']}")
    lines += ["", "## All observed runs (sorted by objective)",
              md_table(pd.DataFrame(packet["observed_runs_sorted_by_objective"])), ""]
    return "\n".join(lines)


def write_prompt_file(packet: dict, path: Path) -> None:
    Path(path).write_text("# SYSTEM PROMPT\n\n" + system_prompt(packet["objective"]["name"]) + "\n\n# USER MESSAGE\n\n" + packet_to_markdown(packet)
                          + "\n\nReview the main GP suggestion as instructed and answer in JSON.\n", encoding="utf-8")


# ------------------------------------------------------------------ the call
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON object in model reply:\n{text[:500]}")
    return json.loads(m.group(0))


IMAGE_CAPTION = ("Figure: the acquisition function projected onto every pair of free knobs (mean over the other "
                 "knobs, context fixed); larger = more worth trying. Dots = measured runs coloured by the objective, "
                 "red star = the suggested next run.")
USER_SUFFIX = "\n\nReview the main GP suggestion as instructed and answer in JSON."
DEFAULT_MODELS = {"anthropic": DEFAULT_MODEL, "openai": "gpt-5"}


def review_messages(packet: dict, image_path: Optional[Path] = None) -> dict:
    """Provider-neutral content of the review request: system prompt, user text, optional PNG bytes."""
    png = Path(image_path).read_bytes() if image_path and Path(image_path).exists() else None
    return {"system": system_prompt(packet["objective"]["name"]),
            "text": (IMAGE_CAPTION + "\n\n" if png else "") + packet_to_markdown(packet) + USER_SUFFIX,
            "image_png": png}


def _call_anthropic(system: str, text: str, png: Optional[bytes], model: str, effort: str = "high"):
    import anthropic
    content = []
    if png:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.standard_b64encode(png).decode("utf-8")}})
    content.append({"type": "text", "text": text})
    client = anthropic.Anthropic()
    kwargs = dict(model=model, max_tokens=16000, system=system, output_config={"effort": effort},
                  messages=[{"role": "user", "content": content}])
    try:
        # server-side refusal fallback (routes a safety decline to another model inside the same call)
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except TypeError:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined the request: {getattr(resp, 'stop_details', None)}")
    reply = "".join(b.text for b in resp.content if b.type == "text")
    return reply, resp.model, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def _call_openai(system: str, text: str, png: Optional[bytes], model: str, effort: str = "high"):
    """OpenAI chat completions with the same system prompt and packet (image as a data URL)."""
    from openai import OpenAI
    content = [{"type": "text", "text": text}]
    if png:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                                   + base64.standard_b64encode(png).decode("utf-8")}})
    resp = OpenAI().chat.completions.create(model=model, messages=[{"role": "system", "content": system},
                                                                   {"role": "user", "content": content}])
    reply = resp.choices[0].message.content or ""
    usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", None),
             "output_tokens": getattr(resp.usage, "completion_tokens", None)}
    return reply, resp.model, usage


PROVIDERS = {"anthropic": _call_anthropic, "openai": _call_openai}


def review(packet: dict, image_path: Optional[Path] = None, model: Optional[str] = None,
           provider: str = "anthropic", effort: str = "high", call=None) -> dict:
    """Ask a model to accept / reject the GP suggestion. Returns the parsed JSON plus raw text.

    provider: "anthropic" (anthropic SDK, needs ANTHROPIC_API_KEY) or "openai" (openai SDK, needs OPENAI_API_KEY).
    call:     any function (system, text, png_bytes_or_None, model) -> (reply_text, model_name, usage_dict)
              for another provider or a local model; it overrides `provider`."""
    provider = (provider or "anthropic").lower()
    if call is None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {list(PROVIDERS)} or pass call=")
        call = PROVIDERS[provider]
        if provider == "anthropic":
            _c = call
            call = lambda s, t, p, m: _c(s, t, p, m, effort)
    model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODEL)
    msg = review_messages(packet, image_path)
    reply, used_model, usage = call(msg["system"], msg["text"], msg["image_png"], model)
    out = _extract_json(reply)
    out["_provider"] = provider
    out["_model"] = used_model
    out["_raw"] = reply
    out["_usage"] = usage
    return out


def review_to_markdown(r: dict, packet: dict) -> str:
    rec = r.get("recommended_recipe", {})
    lines = [f"# LLM review ({r.get('_model', '?')})", "",
             f"**Decision:** {r.get('decision', '?').upper()}   **confidence:** {r.get('confidence', '?')}", "",
             "**Recommended recipe:**", "", "| parameter | value |", "|---|---|"]
    for k in FREE_NAMES:
        if k in rec:
            lines.append(f"| {k} | {rec[k]} |")
    lines += ["", "**Reason:** " + str(r.get("reason", "")), ""]
    if r.get("risks"):
        lines += ["**Risks:**"] + [f"- {x}" for x in r["risks"]] + [""]
    lines += [f"_Tokens: {r.get('_usage')}_"]
    return "\n".join(lines)
