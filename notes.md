# Project Notes

## Original CSV vs ML CSV Results

The original CSV notebooks and ML CSV notebooks produced very different results.

### Original CSV

Source file:

- `data/youtube_shorts_tiktok_trends_2025.csv`

Processed shape:

- Train: `33,655 x 203`
- Validation: `7,212 x 203`
- Test: `7,212 x 203`

Target distribution:

- Roughly balanced across the four classes.
- Each class is close to 25%.

Main results:

- Majority baseline test accuracy: about `0.253`
- Majority baseline macro F1: about `0.101`
- Random Forest validation macro F1: about `0.260`
- Random Forest test macro F1: about `0.249`
- MLP Neural Network test macro F1: about `0.256`

Interpretation:

- The original feature set barely performs above chance.
- Model choice does not make a large difference.
- The weak performance suggests the original features, as currently processed, do not strongly separate the four trend labels.

Important original feature findings:

- Random Forest importance mostly ranked raw engagement features highest.
- Strongest original features included `completion_rate`, `save_rate`, `creator_avg_views`, `like_dislike_ratio`, `dislike_rate`, `avg_watch_time_sec`, `comment_ratio`, `share_rate`, and related engagement fields.
- Feature-group importance was mostly `engagement_observed`, followed by `temporal`, `metadata`, `content_basic`, and `creator`.

## ML CSV Results

Source file:

- `data/youtube_shorts_tiktok_trends_2025.csv_ML.csv`

Processed shape:

- Train: `35,000 x 72`
- Validation: `7,500 x 72`
- Test: `7,500 x 72`

Target distribution:

- More imbalanced than the original CSV.
- `stable`: about 55%
- `rising`: about 25%
- `declining`: about 10%
- `seasonal`: about 9%

Main results:

- Majority baseline test accuracy: about `0.554`
- Majority baseline macro F1: about `0.178`
- Random Forest validation macro F1: about `0.778`
- Random Forest test macro F1: about `0.782`
- MLP Neural Network test accuracy: about `0.816`
- MLP Neural Network test macro F1: about `0.774`

Interpretation:

- The ML CSV gives much stronger predictive performance than the original CSV.
- The improvement is visible in macro F1, not just accuracy, so it is not only caused by the larger `stable` class.
- Random Forest is the best overall model by macro F1.
- MLP is competitive and gives slightly higher test accuracy, but lower macro F1 than Random Forest.

Important ML feature findings:

- Top features included `weekend_hashtag_boost`, `rel_combo`, `share_rate_log`, `rel_share`, `share_hashtag_interaction`, `likes_per_day`, `share_rate`, and `views_per_day`.
- Feature-group importance was mostly `engagement_observed`, then `interactions`, then `metadata`.
- The `interactions` group was very strong as a standalone feature group.

## Dataset Relationship Check

A light inspection suggests the ML CSV is not a simple row-for-row transformation of the original CSV.

Main evidence:

- Original CSV has `48,079` rows.
- ML CSV has `50,000` rows.
- Original CSV has `58` columns.
- ML CSV has `32` columns.
- Only 10 column names overlap directly.
- Shared categorical vocabularies differ.
- Original label distribution is balanced, but ML label distribution is highly skewed toward `stable`.
- Shared numeric fields such as `like_rate` and `share_rate` have very different ranges.
- The ML CSV contains many engineered columns not present in the original, such as log rates, relative rates, and interaction features.

Conclusion:

- The ML CSV should not be treated as a direct preprocessing output from the original CSV.
- It looks like a separately engineered ML-ready dataset, and may be synthetic, modified, or generated from a different process.
- The strong ML results may come from engineered or target-related structure in that file rather than from the original raw dataset.

## Synthetic / Caution Columns

The local data dictionary marks several original fields as synthetic or synthetic-like.

Examples:

- `author_handle`: creator handle/channel, synthetic.
- `trend_duration_days`: days the video remained trending, synthetic.
- `dislikes`: synthetic, platform-aware.
- `sample_comments`: synthetic multilingual comment.
- `title`: synthetic realistic video title.

Other fields to treat carefully:

- `source_hint`
- `notes`
- `row_id`
- `trend_type`
- `engagement_velocity`
- Any feature that describes trend behavior after the trend has already happened.

## No-Synthetic RQ1 Sensitivity Check

An optional RQ1 sensitivity check removed synthetic, synthetic-derived, or cautionary fields from the already processed original-CSV feature set.

Removed fields:

- `dislikes`
- `like_dislike_ratio`
- `dislike_rate`
- `title_length`
- `has_emoji`
- `avg_watch_time_sec`
- `completion_rate`
- `creator_avg_views`

Feature-count change:

- Original processed RQ1 feature count: `203`
- No-synthetic/caution feature count: `195`

Random Forest comparison:

- Original processed validation macro F1: about `0.262`
- No-synthetic/caution validation macro F1: about `0.242`
- Original processed test macro F1: about `0.245`
- No-synthetic/caution test macro F1: about `0.246`

Interpretation:

- Removing the flagged fields lowers validation performance slightly, but test performance is essentially unchanged.
- This suggests the flagged fields contributed some model-usable signal on the validation split, but the benefit does not appear robust on the held-out test split.
- In other words, the synthetic/caution columns may help the model fit one split slightly, but they do not materially improve generalization.
- The original RQ1 finding is therefore not strongly dependent on the synthetic/caution fields.
- The top features after removal are still mostly engagement variables, including `save_rate`, `saves`, `share_rate`, `comment_ratio`, `engagement_comment_rate`, `engagement_share_rate`, `views`, `comments`, and `like_rate`.
- Feature-group importance remains dominated by `engagement_observed`, followed by `temporal` and `metadata`.
- The cleaned version is methodologically safer, but it does not solve the weak predictive performance problem.

## Recommended Direction

The more defensible path is to use the original CSV as the source of truth.

Suggested workflow:

1. Start from `data/youtube_shorts_tiktok_trends_2025.csv`.
2. Remove obvious synthetic, identifier, free-text, and leakage-prone columns.
3. Keep believable observed metadata and engagement fields.
4. Engineer transparent features manually from the original data.
5. Rerun the RQ notebooks on this cleaned and self-engineered version.
6. Treat the ML CSV results only as a comparison or cautionary appendix, not as the main findings.

Safer observed features to keep:

- `platform`
- `country`
- `region`
- `language`
- `category`
- `hashtag`
- `sound_type`
- `duration_sec`
- `views`
- `likes`
- `comments`
- `shares`
- `saves`
- `upload_hour`
- `publish_dayofweek`
- `publish_period`
- `event_season`
- `season`
- `is_weekend`
- `device_type`
- `device_brand`
- `traffic_source`
- `creator_tier`

Features to engineer transparently:

- `like_rate = likes / views`
- `comment_rate = comments / views`
- `share_rate = shares / views`
- `save_rate = saves / views`
- `engagement_total = likes + comments + shares + saves`
- `engagement_per_1k = engagement_total / views * 1000`
- `log_views`
- `log_likes`
- `log_comments`
- `log_shares`
- `log_saves`
- `log_engagement_total`
- `publish_month`
- `publish_day`
- `is_weekend`

Leakage rule of thumb:

- Keep a feature only if it would plausibly be available before or during trend classification.
- Remove a feature if it directly encodes the outcome, describes how long something trended, or was created after observing the target.

## Current Recommendation for Reporting

Main report:

- Use original CSV results, but acknowledge weak predictive performance.
- Explain that raw observed features show limited ability to classify trend labels.
- Discuss engagement features as exploratory associations only.

Improved follow-up analysis:

- Create a cleaned original-data pipeline excluding synthetic and leakage-prone columns.
- Add transparent engineered features from observed counts and dates.
- Compare this cleaned-engineered version against the current original and ML-version results.

ML CSV:

- Keep as a cautionary comparison.
- Mention that it gives strong performance, but its row count, class distribution, vocabularies, and engineered columns do not line up cleanly with the original CSV.
- Avoid using it as the main evidence unless its provenance can be justified.
