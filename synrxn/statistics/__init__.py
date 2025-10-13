"""
statistics_pipeline package

Expose three top-level functions for the three test families:

- :func:`statistics_pipeline.assumptions.run_assumptions`
- :func:`statistics_pipeline.parametric.run_parametric`
- :func:`statistics_pipeline.nonparametric.run_nonparametric`
"""

from .assumptions import run_assumptions  # noqa: F401
from .parametric import run_parametric  # noqa: F401
from .nonparametric import run_nonparametric  # noqa: F401

__all__ = ["run_assumptions", "run_parametric", "run_nonparametric"]
