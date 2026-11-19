#exp 7


import pandas as pd

# 1. Load dataset
df = pd.read_csv("/content/business_analytics_1000_rows.csv")

# 2. Create target
df["satisfaction_class"] = (
    df["customer_satisfaction_1_to_5"] >= 4).astype(int)

# 3. Select features
drop = ["order_id", "customer_id", "order_date", "currency",
        "customer_satisfaction_1_to_5", "satisfaction_class"]

X = df.drop(columns=drop)
y = df["satisfaction_class"]

# 4. Encoding
from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in X.select_dtypes(include="object"):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 5. Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200, max_depth=8,
    random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 7. Prediction and evaluation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["Low", "High"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 8. Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
print("Top 10 Features:\n", importance.sort_values(ascending=False).head(10))

# 9. Baseline comparison
baseline = max(y_test.mean(), 1 - y_test.mean())
accuracy = accuracy_score(y_test, pred)

print("Baseline Accuracy:", baseline)
print("Model Improvement:", (accuracy - baseline) * 100)