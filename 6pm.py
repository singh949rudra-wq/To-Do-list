#exp 6

import numpy as np
import pandas as pd
import joblib

# 1. Load dataset
df = pd.read_csv('/content/financial_fraud_dataset.csv').drop(columns=['transaction_id'])
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# 2. Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('enc', OneHotEncoder(handle_unknown='ignore'),
                         X.select_dtypes(exclude=[np.number]).columns)],
                       remainder='passthrough')
X = ct.fit_transform(X)
X = X.toarray() if hasattr(X, 'toarray') else X

# 3. Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

# 4. Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 5. Model selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

models = {
    'LR': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RF': RandomForestClassifier(class_weight='balanced'),
    'GB': GradientBoostingClassifier()
}

scores = {n: cross_val_score(m, X_train, y_train, cv=5).mean()
          for n, m in models.items()}
best = max(scores, key=scores.get)
model = models[best].fit(X_train, y_train)

# 6. Prediction
pred = model.predict(X_test)

# 7. Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report

print("Best Model:", best)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1 Score:", f1_score(y_test, pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))

# 8. Actual vs Predicted
result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result.to_csv('Fraud_Actual_vs_Predicted.csv', index=False)
print(result.head(20))

# 9. Save model
joblib.dump({'model': model, 'scaler': sc, 'encoder': ct,
             'model_name': best}, 'fraud_detection_model.pkl')
print("Model saved!")