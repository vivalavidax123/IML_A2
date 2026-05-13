import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("youtube_shorts_tiktok_trends_2025.csv_ML.csv")

# %% Platform vs Trend Label
platform_trend = pd.crosstab(
    df["platform"],
    df["trend_label"],
    normalize="index"
) * 100

print(platform_trend)

platform_trend.plot(kind="bar", stacked=True)

plt.title("Trend Label Distribution by Platform")
plt.ylabel("Percentage")
plt.xlabel("Platform")

plt.legend(title="Trend Label")
plt.tight_layout()

plt.show()

# %% Category vs Trend Label
category_trend = pd.crosstab(
    df["category"],
    df["trend_label"],
    normalize="index"
) * 100

print(category_trend)

category_trend.plot(kind="bar", stacked=True, figsize=(12,6))


plt.title("Trend Label Distribution by Category")
plt.ylabel("Percentage")
plt.xlabel("Category")

plt.legend(title="Trend Label")
plt.tight_layout()

plt.show()

# %% Traffic Source vs Trend Label
traffic_trend = pd.crosstab(
    df["traffic_source"],
    df["trend_label"],
    normalize="index"
) * 100

print(traffic_trend)

traffic_trend.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("Trend Label Distribution by Traffic Source")
plt.ylabel("Percentage")
plt.xlabel("Traffic Source")

plt.legend(title="Trend Label")
plt.tight_layout()

plt.show()

# %% Creator Tier vs Trend Label
creator_trend = pd.crosstab(
    df["creator_tier"],
    df["trend_label"],
    normalize="index"
) * 100

print(creator_trend)

creator_trend.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("Trend Label Distribution by Creator Tier")
plt.ylabel("Percentage")
plt.xlabel("Creator Tier")

plt.legend(title="Trend Label")
plt.tight_layout()

plt.show()

# %% Prepare the data
data = df.copy()

le = LabelEncoder()
data["trend_label"] = le.fit_transform(data["trend_label"])

X = data.drop(columns = [
    "trend_label",
    "platform",
    "region",
    "language",
    "category",
    "traffic_source",
    "device_brand",
    "creator_tier"
])

y = data["trend_label"]

print(X.head())
print(y.head())

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, y_train)

print("Random Forest trained successfully")

# %% Extract Feature Importantce
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(15))

# %% Plot Top feature Importance
importance_top = importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    importance_top["Feature"],
    importance_top["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# %% Model evaluation
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

# %%
