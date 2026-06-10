# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("data/processed/crop_ml_ready.csv")

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------
# TRAIN TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# MODEL 1 - Logistic Regression
# -----------------------
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# -----------------------
# MODEL 2 - Decision Tree
# -----------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# -----------------------
# MODEL 3 - Random Forest (BEST)
# -----------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -----------------------
# PRINT RESULTS
# -----------------------
print("Logistic Regression:", lr_acc)
print("Decision Tree:", dt_acc)
print("Random Forest:", rf_acc)

# -----------------------
# SAVE BEST MODEL
# -----------------------
model_path = os.path.join("models", "crop_model.pkl")

os.makedirs("models", exist_ok=True)

joblib.dump(rf, model_path)
print("Model saved successfully!")