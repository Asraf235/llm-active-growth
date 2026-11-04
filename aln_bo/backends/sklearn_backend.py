"""scikit-learn backend: thin re-export of the numpy implementation.

The actual code lives in aln_bo/surrogate.py (GP, GPList) and
aln_bo/acquisition.py (EI, MC-EHVI, optimiser). This module only gives it the
uniform backend interface described in aln_bo/backends/__init__.py.
"""
from ..surrogate import GP, GPList                     # noqa: F401
from ..acquisition import suggest_ei, suggest_ehvi     # noqa: F401

NAME = "sklearn"
