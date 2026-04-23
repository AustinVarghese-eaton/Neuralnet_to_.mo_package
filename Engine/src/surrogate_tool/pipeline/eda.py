from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Force headless-safe backend (important when running without GUI context)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from surrogate_tool.contracts.run_config import load_run_config
from surrogate_tool.contracts.status import write_status
from surrogate_tool.paths import runs_root

sns.set_theme(style="whitegrid")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Structured summary:
      - shape, columns, dtypes
      - missing values
      - duplicate rows
      - numeric summary (df.describe numeric)
    Mirrors your original eda.py intent/output. [1](https://eaton-my.sharepoint.com/personal/austinvarghese_eaton_com/_layouts/15/Doc.aspx?sourcedoc=%7B832DAC3D-AEC5-4FF8-AF5F-6147B78FCC6B%7D&file=NN_2.docx&action=default&mobileredirect=true)
    """
    profile = {
        "timestamp_utc": _utc_now(),
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    desc = df.describe(include=[np.number]).T
    profile["numeric_summary"] = desc.round(6).to_dict()
    return profile


def plot_distributions(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    dist_<col>.png for numeric columns. [1](https://eaton-my.sharepoint.com/personal/austinvarghese_eaton_com/_layouts/15/Doc.aspx?sourcedoc=%7B832DAC3D-AEC5-4FF8-AF5F-6147B78FCC6B%7D&file=NN_2.docx&action=default&mobileredirect=true)
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution: {col}")
            plt.tight_layout()
            plt.savefig(figures_dir / f"dist_{col}.png", dpi=160)
            plt.close()


def plot_pairwise(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    pairplot_sampled.png (sample up to 400 rows). [1](https://eaton-my.sharepoint.com/personal/austinvarghese_eaton_com/_layouts/15/Doc.aspx?sourcedoc=%7B832DAC3D-AEC5-4FF8-AF5F-6147B78FCC6B%7D&file=NN_2.docx&action=default&mobileredirect=true)
    """
    sample = df.sample(min(len(df), 400), random_state=42)
    g = sns.pairplot(sample, corner=True, diag_kind="hist")
    g.fig.suptitle("Pairplot (sampled)", y=1.02)
    g.savefig(figures_dir / "pairplot_sampled.png", dpi=160)
    plt.close("all")


def plot_correlation(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    correlation_heatmap.png using numeric-only correlation. [1](https://eaton-my.sharepoint.com/personal/austinvarghese_eaton_com/_layouts/15/Doc.aspx?sourcedoc=%7B832DAC3D-AEC5-4FF8-AF5F-6147B78FCC6B%7D&file=NN_2.docx&action=default&mobileredirect=true)
    """
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def plot_outliers_box(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    box_<col>.png for numeric columns. [1](https://eaton-my.sharepoint.com/personal/austinvarghese_eaton_com/_layouts/15/Doc.aspx?sourcedoc=%7B832DAC3D-AEC5-4FF8-AF5F-6147B78FCC6B%7D&file=NN_2.docx&action=default&mobileredirect=true)
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(7, 3))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot: {col}")
            plt.tight_layout()
            plt.savefig(figures_dir / f"box_{col}.png", dpi=160)
            plt.close()


def run_eda(run_id: str, attempt_num: int) -> dict:
    """
    Runs EDA inside:
      runs/<run_id>/attempts/attempt_<N>/reports/
    Prefers cleaned dataset:
      attempts/attempt_<N>/processed/cleaned_dataset.csv
    Writes:
      - eda_profile.json
      - figures/*.png
      - pairplot_error.json if pairplot fails
    """
    run_dir = runs_root() / run_id
    cfg_path = run_dir / "input" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {cfg_path}")

    cfg = load_run_config(cfg_path)

    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_num:03d}"
    processed_dir = attempt_dir / "processed"
    reports_dir = attempt_dir / "reports"
    figures_dir = reports_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    write_status(status_path, state="EDA", message="Loading dataset for EDA...", progress=0.1)

    cleaned_csv = processed_dir / "cleaned_dataset.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {cleaned_csv}\n"
            f"Run preprocess first:\n"
            f"  python -m surrogate_tool preprocess --run-id {run_id} --attempt {attempt_num} --inputs ... --outputs ..."
        )

    df = pd.read_csv(cleaned_csv)

    write_status(status_path, state="EDA", message="Building EDA profile and figures...", progress=0.4)

    profile = profile_dataframe(df)
    profile.update({
        "run_id": cfg.run_id,
        "attempt": f"attempt_{attempt_num:03d}",
        "dataset_used": str(cleaned_csv),
        "package_name": cfg.package_name,
    })

    (reports_dir / "eda_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # Figures
    plot_distributions(df, figures_dir)
    plot_correlation(df, figures_dir)
    plot_outliers_box(df, figures_dir)

    # Optional heavier plot
    try:
        plot_pairwise(df, figures_dir)
    except Exception as e:
        (reports_dir / "pairplot_error.json").write_text(
            json.dumps({"pairplot_error": str(e)}, indent=2),
            encoding="utf-8"
        )

    write_status(status_path, state="EDA_DONE", message="EDA complete.", progress=1.0)
    return {
        "eda_profile": str(reports_dir / "eda_profile.json"),
        "figures_dir": str(figures_dir),
        "cleaned_csv": str(cleaned_csv),
    }
