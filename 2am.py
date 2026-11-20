import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Load & Impute
dataset = pd.read_csv('network_intrusion_dataset-Lab01.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X[:, 1:6] = SimpleImputer(strategy='mean').fit_transform(X[:, 1:6])

# 2. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 3. Model (Swap any of the 15 here)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# 4. Evaluate
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

tn, fp, fn, tp = cm.ravel()
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity (TPR) : {tp/(tp+fn):.4f}")
print(f"Specificity (TNR) : {tn/(tn+fp):.4f}")
print(f"Precision (PPV)   : {tp/(tp+fp):.4f}")
print(f"NPV               : {tn/(tn+fn):.4f}")
print(f"FPR               : {fp/(fp+tn):.4f}")
print(f"FNR               : {fn/(tp+fn):.4f}")
print(f"FDR               : {fp/(tp+fp):.4f}")
print(f"Overall Accuracy  : {(tp+tn)/(tp+fp+fn+tn):.4f}")