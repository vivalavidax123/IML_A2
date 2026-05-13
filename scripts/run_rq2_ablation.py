"""Run RQ2 feature-group ablation experiments.

RQ2 asks how different feature groups affect trend-label prediction. This
script keeps the experiment reproducible from the command line and writes the
tables/figures needed for the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "youtube_shorts_tiktok_trends_2025.csv"
ASSET_DIR = PROJECT_ROOT / "report" / "assets"
DRAFT_DIR = PROJECT_ROOT / "report" / "drafts"
FIGURE_DIR = PROJECT_ROOT / "figures"

TARGET = "trend_label"
RANDOM_STATE = 42

TEXT_ID_OR_NOTES_COLUMNS = [
    "row_id",
    "source_hint",
    "notes",
    "title",
    "title_keywords",
    "tags",
    "sample_comments",
    "author_handle",
    "music_track",
    "publish_date_approx",
    "year_month",
]

POSSIBLE_LEAKAGE_COLUMNS = [
    "trend_duration_days",
    "trend_type",
    "engagement_velocity",
]

FEATURE_GROUPS = {
    "platform_metadata": [
        "platform",
        "country",
        "region",
        "language",
        "category",
        "hashtag",
        "sound_type",
        "device_type",
        "device_brand",
        "traffic_source",
    ],
    "temporal": [
        "week_of_year",
        "upload_hour",
        "publish_dayofweek",
        "publish_period",
        "event_season",
        "season",
        "is_weekend",
        "publish_month",
        "publish_day",
    ],
    "content_basic": [
        "genre",
        "duration_sec",
        "title_length",
        "has_emoji",
    ],
    "creator": [
        "creator_avg_views",
        "creator_tier",
    ],
    "engagement_observed": [
        "views",
        "likes",
        "comments",
        "shares",
        "saves",
        "dislikes",
        "engagement_total",
        "engagement_rate",
        "comment_ratio",
        "share_rate",
        "save_rate",
        "like_dislike_ratio",
        "like_rate",
        "dislike_rate",
        "engagement_per_1k",
        "engagement_like_rate",
        "engagement_comment_rate",
        "engagement_share_rate",
        "avg_watch_time_sec",
        "completion_rate",
    ],
}


def make_encoder() -> OneHotEncoder:
    """Create a version-tolerant OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_clean_data() -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    df = pd.read_csv(DATA_PATH).drop_duplicates().copy()

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip().str.lower()

    df["publish_date_approx"] = pd.to_datetime(df["publish_date_approx"], errors="coerce")
    df["publish_month"] = df["publish_date_approx"].dt.month
    df["publish_day"] = df["publish_date_approx"].dt.day

    drop_columns = [TARGET] + TEXT_ID_OR_NOTES_COLUMNS + POSSIBLE_LEAKAGE_COLUMNS
    drop_columns = [column for column in drop_columns if column in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[TARGET]

    groups = {
        group_name: [column for column in columns if column in X.columns]
        for group_name, columns in FEATURE_GROUPS.items()
    }
    non_engagement = sorted(
        set(
            groups["platform_metadata"]
            + groups["temporal"]
            + groups["content_basic"]
            + groups["creator"]
        )
    )
    groups["non_engagement_combined"] = non_engagement
    groups["all_features"] = sorted(set(non_engagement + groups["engagement_observed"]))

    return X, y, groups


def build_pipeline(X_train: pd.DataFrame, estimator) -> Pipeline:
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X_train.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def evaluate(y_true: pd.Series, y_pred, model: str, feature_group: str) -> dict[str, float | str]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "model": model,
        "feature_group": feature_group,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def plot_feature_comparison(results: pd.DataFrame, output_path: Path) -> None:
    best_by_group = (
        results.sort_values("f1_macro", ascending=False)
        .groupby("feature_group", as_index=False)
        .first()
        .sort_values("f1_macro", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(best_by_group["feature_group"], best_by_group["f1_macro"], color="#3b82f6")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("RQ2 Best Macro F1 by Feature Group")
    ax.tick_params(axis="x", rotation=25)
    for index, value in enumerate(best_by_group["f1_macro"]):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true: pd.Series, y_pred, labels: list[str], title: str, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_groups = load_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    result_rows = []
    prediction_cache = {}
    for group_name, columns in feature_groups.items():
        X_train_group = X_train[columns]
        X_test_group = X_test[columns]
        for model_name, estimator in models.items():
            pipeline = build_pipeline(X_train_group, estimator)
            pipeline.fit(X_train_group, y_train)
            predictions = pipeline.predict(X_test_group)
            result_rows.append(evaluate(y_test, predictions, model_name, group_name))
            prediction_cache[(group_name, model_name)] = predictions

    results = pd.DataFrame(result_rows).sort_values(
        ["f1_macro", "accuracy"],
        ascending=False,
    )
    results_path = ASSET_DIR / "rq2_feature_ablation_results.csv"
    groups_path = ASSET_DIR / "rq2_feature_groups.csv"
    plot_path = ASSET_DIR / "rq2_feature_ablation_f1.png"
    summary_path = DRAFT_DIR / "rq2_results_summary.md"

    results.to_csv(results_path, index=False)
    pd.DataFrame(
        [
            {"feature_group": group_name, "feature": feature}
            for group_name, features in feature_groups.items()
            for feature in features
        ]
    ).to_csv(groups_path, index=False)
    plot_feature_comparison(results, plot_path)

    best = results.iloc[0]
    best_by_group = (
        results.sort_values("f1_macro", ascending=False)
        .groupby("feature_group", as_index=False)
        .first()
        .sort_values("f1_macro", ascending=False)
    )

    labels = sorted(y.unique())
    best_predictions = prediction_cache[(best["feature_group"], best["model"])]
    confusion_path = ASSET_DIR / f"rq2_confusion_{best['feature_group']}_{best['model']}.png"
    plot_confusion(
        y_test,
        best_predictions,
        labels=labels,
        title=f"RQ2 Best Model: {best['feature_group']} / {best['model']}",
        output_path=confusion_path,
    )

    summary_lines = [
        "# RQ2 Feature Group Ablation Summary",
        "",
        f"Dataset: `{DATA_PATH.relative_to(PROJECT_ROOT)}`",
        f"Rows after duplicate removal: {len(X):,}",
        f"Target: `{TARGET}`",
        f"Labels: {', '.join(labels)}",
        "",
        "## Feature Groups",
        "",
    ]
    for group_name, features in feature_groups.items():
        summary_lines.append(f"- `{group_name}` ({len(features)} features): {', '.join(features)}")

    summary_lines.extend(
        [
            "",
            "## Best Result",
            "",
            (
                f"Best configuration: `{best['model']}` with `{best['feature_group']}` "
                f"features, macro F1 = {best['f1_macro']:.3f}, "
                f"accuracy = {best['accuracy']:.3f}."
            ),
            "",
            "## Best Model Per Feature Group",
            "",
            best_by_group.to_markdown(index=False),
            "",
            "## Generated Assets",
            "",
            f"- Results table: `{results_path.relative_to(PROJECT_ROOT)}`",
            f"- Feature list: `{groups_path.relative_to(PROJECT_ROOT)}`",
            f"- Comparison plot: `{plot_path.relative_to(PROJECT_ROOT)}`",
            f"- Best confusion matrix: `{confusion_path.relative_to(PROJECT_ROOT)}`",
            "",
            "## Report Notes",
            "",
            "- Use macro F1 as the primary metric because the report compares four trend classes.",
            "- Treat `engagement_observed` as observed-performance features, not early-prediction features.",
            "- Compare `non_engagement_combined` against `all_features` to discuss the value of engagement signals.",
        ]
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote {results_path}")
    print(f"Wrote {groups_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {summary_path}")
    print()
    print(best_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
