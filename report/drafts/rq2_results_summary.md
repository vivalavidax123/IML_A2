# RQ2 Feature Group Ablation Summary

Dataset: `data/youtube_shorts_tiktok_trends_2025.csv`
Rows after duplicate removal: 48,079
Target: `trend_label`
Labels: declining, rising, seasonal, stable

## Feature Groups

- `platform_metadata` (10 features): platform, country, region, language, category, hashtag, sound_type, device_type, device_brand, traffic_source
- `temporal` (9 features): week_of_year, upload_hour, publish_dayofweek, publish_period, event_season, season, is_weekend, publish_month, publish_day
- `content_basic` (4 features): genre, duration_sec, title_length, has_emoji
- `creator` (2 features): creator_avg_views, creator_tier
- `engagement_observed` (20 features): views, likes, comments, shares, saves, dislikes, engagement_total, engagement_rate, comment_ratio, share_rate, save_rate, like_dislike_ratio, like_rate, dislike_rate, engagement_per_1k, engagement_like_rate, engagement_comment_rate, engagement_share_rate, avg_watch_time_sec, completion_rate
- `non_engagement_combined` (25 features): category, country, creator_avg_views, creator_tier, device_brand, device_type, duration_sec, event_season, genre, has_emoji, hashtag, is_weekend, language, platform, publish_day, publish_dayofweek, publish_month, publish_period, region, season, sound_type, title_length, traffic_source, upload_hour, week_of_year
- `all_features` (45 features): avg_watch_time_sec, category, comment_ratio, comments, completion_rate, country, creator_avg_views, creator_tier, device_brand, device_type, dislike_rate, dislikes, duration_sec, engagement_comment_rate, engagement_like_rate, engagement_per_1k, engagement_rate, engagement_share_rate, engagement_total, event_season, genre, has_emoji, hashtag, is_weekend, language, like_dislike_ratio, like_rate, likes, platform, publish_day, publish_dayofweek, publish_month, publish_period, region, save_rate, saves, season, share_rate, shares, sound_type, title_length, traffic_source, upload_hour, views, week_of_year

## Best Result

Best configuration: `random_forest` with `all_features` features, macro F1 = 0.255, accuracy = 0.255.

## Best Model Per Feature Group

| feature_group           | model         |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|:------------------------|:--------------|-----------:|------------------:|---------------:|-----------:|
| all_features            | random_forest |   0.255408 |          0.256164 |       0.255387 |   0.255246 |
| platform_metadata       | random_forest |   0.254888 |          0.25516  |       0.254804 |   0.254826 |
| creator                 | random_forest |   0.2526   |          0.25264  |       0.252729 |   0.252563 |
| content_basic           | random_forest |   0.252184 |          0.252318 |       0.252246 |   0.252119 |
| non_engagement_combined | random_forest |   0.252184 |          0.252177 |       0.252131 |   0.251709 |
| temporal                | linear_svm    |   0.25052  |          0.250116 |       0.250016 |   0.247071 |
| engagement_observed     | random_forest |   0.242408 |          0.242325 |       0.242325 |   0.242275 |

## Generated Assets

- Results table: `report/assets/rq2_feature_ablation_results.csv`
- Feature list: `report/assets/rq2_feature_groups.csv`
- Comparison plot: `report/assets/rq2_feature_ablation_f1.png`
- Best confusion matrix: `report/assets/rq2_confusion_all_features_random_forest.png`

## Report Notes

- Use macro F1 as the primary metric because the report compares four trend classes.
- Treat `engagement_observed` as observed-performance features, not early-prediction features.
- Compare `non_engagement_combined` against `all_features` to discuss the value of engagement signals.