import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart_Disease_Prediction")

def train_challenger():
    print("1. Loading versioned dataset snapshot...")
    df = pd.read_parquet("data/processed/patient_records_v1.parquet")

    categorical_cols = ["sex", "chest_pain_type", "smoker_status"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=["patient_id", "has_heart_disease"])
    y = df_encoded["has_heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    params = {"C": 1.0, "max_iter": 1000, "random_state": 42}

    print("2. Training CHALLENGER Model (Logistic Regression)...")
    with mlflow.start_run(run_name="Challenger_LogisticRegression") as run:
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"   --> Accuracy:  {acc:.4f}")
        print(f"   --> ROC-AUC:   {auc:.4f}")
        print(f"   --> F1 Score:  {f1:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        joblib.dump(scaler, "models/scaler.pkl")
        mlflow.log_artifact("models/scaler.pkl")

        joblib.dump(model, "models/challenger_model.pkl")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="HeartDisease_Challenger"
        )
        print("✅ CHALLENGER Model logged and registered in MLflow!")

if __name__ == "__main__":
    train_challenger()