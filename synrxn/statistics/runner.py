"""
Runner module: a simple CLI and demo data generator.

Example
-------

.. code-block:: console

    $ python -m statistics_pipeline.runner --csv path/to/scoring.csv --save-root statistics

Programmatic usage:

.. code-block:: python

    >>> from statistics_pipeline.runner import main
    >>> summary = main(csv_path=None, save_root="statistics_demo")
    >>> "assumptions" in summary
    True
"""

from typing import Dict, Optional

import argparse
import pandas as pd
import numpy as np

from .assumptions import run_assumptions
from .parametric import run_parametric
from .nonparametric import run_nonparametric


def _demo_df(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "mcc"]
    methods = ["GINE_Classifier", "NoCross_Classifier", "SVM_Classifier"]
    cv_cycles = list(range(1, 11))
    rows = []
    for met in metrics:
        for cv in cv_cycles:
            row = {"scoring": met, "cv_cycle": cv}
            for m in methods:
                base = 0.90 if met == "accuracy" else 0.85
                noise = rng.normal(0, 0.02)
                row[m] = base + noise + (0.01 if "GINE" in m else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def main(
    csv_path: Optional[str] = None, save_root: str = "statistics"
) -> Dict[str, object]:
    """
    Run the full pipeline (assumptions, parametric, nonparametric) and return a summary.

    :param csv_path: Optional path to a scoring CSV (wide format: 'scoring','cv_cycle',<methods...>).
                     If None, a synthetic demo dataset is used.
    :type csv_path: Optional[str]
    :param save_root: Root folder to store outputs (default 'statistics').
    :type save_root: str
    :returns: Summary dict describing produced directories.
    :rtype: Dict[str, object]

    .. code-block:: python

        >>> # demo run (uses synthetic data)
        >>> summary = main(csv_path=None, save_root="statistics_demo")
        >>> isinstance(summary, dict)
        True
    """
    df = pd.read_csv(csv_path) if csv_path else _demo_df()

    summary = {}
    summary["assumptions"] = run_assumptions(df, save_root=save_root)
    summary["parametric"] = run_parametric(
        df,
        save_root=save_root,
        direction_dict={"accuracy": "maximize", "mcc": "maximize"},
        effect_dict={"accuracy": 0.05, "mcc": 0.05},
        left_xlim=-0.15,
        right_xlim=0.15,
    )
    summary["nonparametric"] = run_nonparametric(df, save_root=save_root)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run split statistics pipelines.")
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to scoring CSV (optional)."
    )
    parser.add_argument(
        "--save-root", type=str, default="statistics", help="Root output folder."
    )
    args = parser.parse_args()
    out = main(csv_path=args.csv, save_root=args.save_root)
    print("Saved outputs summary:", out)
