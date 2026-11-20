import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

# 1. Load data
dataset = pd.read_csv("agri_soil_crop_dataset_1000.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Encode & Impute
le = LabelEncoder()
X[:, 11] = le.fit_transform(X[:, 11])
X[:, 12] = le.fit_transform(X[:, 12])

le_class = LabelEncoder()
y = le_class.fit_transform(y)

X[:, 1:11] = SimpleImputer(strategy='mean').fit_transform(X[:, 1:11])

# 3. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Model (Swap any of the 15 here)
classifier = LGBMClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# 5. Evaluate
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

print("\nPer-Class Accuracy:")
for crop, acc in zip(le_class.classes_, cm.diagonal() / cm.sum(axis=1)):
    print(f"{crop:12s} : {acc:.4f}")