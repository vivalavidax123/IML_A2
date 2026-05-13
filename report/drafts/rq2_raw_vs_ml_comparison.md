# RQ2 Raw CSV vs ML CSV Comparison

## Best Overall Result

| dataset   | feature_group   | model         |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|:----------|:----------------|:--------------|-----------:|------------------:|---------------:|-----------:|
| ml_csv    | combined        | random_forest |   0.8199   |          0.808259 |       0.764287 |   0.782812 |
| raw_csv   | all_features    | random_forest |   0.255408 |          0.256164 |       0.255387 |   0.255246 |

## Best Model Per Dataset And Feature Group

| dataset   | feature_group           | model               |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|:----------|:------------------------|:--------------------|-----------:|------------------:|---------------:|-----------:|
| ml_csv    | combined                | random_forest       |   0.8199   |          0.808259 |       0.764287 |   0.782812 |
| ml_csv    | content_engagement      | random_forest       |   0.8143   |          0.801264 |       0.756271 |   0.775754 |
| ml_csv    | platform_context        | logistic_regression |   0.4744   |          0.427071 |       0.497427 |   0.420055 |
| ml_csv    | creator                 | linear_svm          |   0.563    |          0.280002 |       0.269195 |   0.223891 |
| raw_csv   | all_features            | random_forest       |   0.255408 |          0.256164 |       0.255387 |   0.255246 |
| raw_csv   | platform_metadata       | random_forest       |   0.254888 |          0.25516  |       0.254804 |   0.254826 |
| raw_csv   | creator                 | random_forest       |   0.2526   |          0.25264  |       0.252729 |   0.252563 |
| raw_csv   | content_basic           | random_forest       |   0.252184 |          0.252318 |       0.252246 |   0.252119 |
| raw_csv   | non_engagement_combined | random_forest       |   0.252184 |          0.252177 |       0.252131 |   0.251709 |
| raw_csv   | temporal                | linear_svm          |   0.25052  |          0.250116 |       0.250016 |   0.247071 |
| raw_csv   | engagement_observed     | random_forest       |   0.242408 |          0.242325 |       0.242325 |   0.242275 |

## Interpretation For Report

The ML-ready CSV achieved a much higher best macro F1 (0.783) than the raw CSV (0.255), a gap of 0.528.

This gap is mainly caused by the ML CSV's engineered engagement and relative-performance features, such as `rel_like`, `rel_share`, `rel_combo`, `views_per_day`, and `likes_per_day`. These features are closer to the trend label than ordinary raw metadata.

For the report, use the raw CSV result to show the limitation of strict metadata-based trend prediction, and use the ML CSV result to discuss feature-group effectiveness when processed engagement signals are available.

## Generated Assets

- Comparison table: `report/assets/rq2_raw_vs_ml_comparison.csv`
- Comparison plot: `report/assets/rq2_raw_vs_ml_best_f1.png`