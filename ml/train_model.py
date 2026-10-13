import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def train_and_evaluate():
    df = pd.read_csv("ml/dataset.csv")
    df = pd.get_dummies(df, columns=["task_type"])

    features = [c for c in df.columns if c not in ["optimal_device", "cpu_time_us", "gpu_time_us"]]
    X = df[features]
    y = df["optimal_device"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(f"Classifier Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    os.makedirs("ml/models", exist_ok=True)
    joblib.dump(clf, "ml/models/rf_scheduler.pkl")
    print("Model serialized to ml/models/rf_scheduler.pkl")


if __name__ == "__main__":
    train_and_evaluate()
