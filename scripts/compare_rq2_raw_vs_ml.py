"""Compare RQ2 raw-CSV and ML-CSV ablation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "report" / "assets"
DRAFT_DIR = PROJECT_ROOT / "report" / "drafts"

RAW_RESULTS_PATH = ASSET_DIR / "rq2_feature_ablation_results.csv"
ML_RESULTS_PATH = ASSET_DIR / "rq2_ml_feature_ablation_results.csv"


def read_results(path: Path, dataset: str) -> pd.DataFrame:
    results = pd.read_csv(path)
    results["dataset"] = dataset
    return results


def best_by_dataset_group(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["dataset", "feature_group", "f1_macro", "accuracy"], ascending=[True, True, False, False])
        .groupby(["dataset", "feature_group"], as_index=False)
        .first()
        .sort_values(["dataset", "f1_macro"], ascending=[True, False])
    )


def plot_best_overall(comparison: pd.DataFrame, output_path: Path) -> None:
    best_overall = (
        comparison.sort_values("f1_macro", ascending=False)
        .groupby("dataset", as_index=False)
        .first()
        .sort_values("dataset")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(best_overall["dataset"], best_overall["f1_macro"], color=["#3b82f6", "#16a34a"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Best Macro F1")
    ax.set_title("RQ2 Raw vs ML CSV")
    for index, row in enumerate(best_overall.itertuples(index=False)):
        ax.text(index, row.f1_macro + 0.015, f"{row.f1_macro:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    raw_results = read_results(RAW_RESULTS_PATH, "raw_csv")
    ml_results = read_results(ML_RESULTS_PATH, "ml_csv")
    all_results = pd.concat([raw_results, ml_results], ignore_index=True)
    comparison = best_by_dataset_group(all_results)

    comparison_path = ASSET_DIR / "rq2_raw_vs_ml_comparison.csv"
    plot_path = ASSET_DIR / "rq2_raw_vs_ml_best_f1.png"
    summary_path = DRAFT_DIR / "rq2_raw_vs_ml_comparison.md"

    comparison.to_csv(comparison_path, index=False)
    plot_best_overall(comparison, plot_path)

    best_overall = (
        comparison.sort_values("f1_macro", ascending=False)
        .groupby("dataset", as_index=False)
        .first()
        .sort_values("dataset")
    )
    raw_best = best_overall[best_overall["dataset"] == "raw_csv"].iloc[0]
    ml_best = best_overall[best_overall["dataset"] == "ml_csv"].iloc[0]
    f1_gap = ml_best["f1_macro"] - raw_best["f1_macro"]

    summary = [
        "# RQ2 Raw CSV vs ML CSV Comparison",
        "",
        "## Best Overall Result",
        "",
        best_overall.to_markdown(index=False),
        "",
        "## Best Model Per Dataset And Feature Group",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interpretation For Report",
        "",
        (
            f"The ML-ready CSV achieved a much higher best macro F1 "
            f"({ml_best['f1_macro']:.3f}) than the raw CSV ({raw_best['f1_macro']:.3f}), "
            f"a gap of {f1_gap:.3f}."
        ),
        "",
        (
            "This gap is mainly caused by the ML CSV's engineered engagement and "
            "relative-performance features, such as `rel_like`, `rel_share`, "
            "`rel_combo`, `views_per_day`, and `likes_per_day`. These features are "
            "closer to the trend label than ordinary raw metadata."
        ),
        "",
        (
            "For the report, use the raw CSV result to show the limitation of "
            "strict metadata-based trend prediction, and use the ML CSV result to "
            "discuss feature-group effectiveness when processed engagement signals "
            "are available."
        ),
        "",
        "## Generated Assets",
        "",
        f"- Comparison table: `{comparison_path.relative_to(PROJECT_ROOT)}`",
        f"- Comparison plot: `{plot_path.relative_to(PROJECT_ROOT)}`",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"Wrote {comparison_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    print()
    print(best_overall.to_string(index=False))


if __name__ == "__main__":
    main()
