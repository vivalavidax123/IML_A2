# RQ2 ML Feature Group Ablation Summary

Dataset: `data/youtube_shorts_tiktok_trends_2025.csv_ML.csv`
Rows: 50,000
Target: `trend_label`
Labels: declining, rising, seasonal, stable

## Best Result

Best configuration: `random_forest` with `combined` features, macro F1 = 0.783, accuracy = 0.820.

## Best Model Per Feature Group

| feature_group      | model               |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|:-------------------|:--------------------|-----------:|------------------:|---------------:|-----------:|
| combined           | random_forest       |     0.8199 |          0.808259 |       0.764287 |   0.782812 |
| content_engagement | random_forest       |     0.8143 |          0.801264 |       0.756271 |   0.775754 |
| platform_context   | logistic_regression |     0.4744 |          0.427071 |       0.497427 |   0.420055 |
| creator            | linear_svm          |     0.563  |          0.280002 |       0.269195 |   0.223891 |

## Generated Assets

- Results table: `report/assets/rq2_ml_feature_ablation_results.csv`
- Feature list: `report/assets/rq2_ml_feature_groups.csv`
- Comparison plot: `report/assets/rq2_ml_feature_ablation_f1.png`
- Best confusion matrix: `report/assets/rq2_ml_confusion_combined_random_forest.png`

## Report Notes

- The ML CSV contains engineered engagement and relative-performance features.
- These features make the task closer to observed trend classification than strict pre-publication prediction.