# Imports (#1 to #15 in order)
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# CatBoost #1
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# DTC #2
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# GBC #3
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# KNeighborsClassifier #4
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# LDA #5
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# LG #6
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# LGBM #7
classifier = LGBMClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# SVM #8
classifier = SVC(kernel='linear', probability=True, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# MLPC #9
classifier = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# NB #10
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# QDA #11
classifier = QuadraticDiscriminantAnalysis()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# RFC #12
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# XGB #13
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# AdaBoostClassifier #14
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# SVM #15
classifier = SVC(kernel='rbf', probability=True, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)