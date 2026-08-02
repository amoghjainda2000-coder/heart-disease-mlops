import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart_Disease_Prediction")

def train_champion():
    print("1. Loading versioned dataset snapshot...")
    df = pd.read_parquet("data/processed/patient_records_v1.parquet")

    # Feature selection & encoding
    categorical_cols = ["sex", "chest_pain_type", "smoker_status"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=["patient_id", "has_heart_disease"])
    y = df_encoded["has_heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }

    print("2. Training CHAMPION Model (Random Forest)...")
    with mlflow.start_run(run_name="Champion_RandomForest") as run:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions & Evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"   --> Accuracy:  {acc:.4f}")
        print(f"   --> ROC-AUC:   {auc:.4f}")
        print(f"   --> F1 Score:  {f1:.4f}")

        # Log parameters & metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log feature list artifact for FastAPI
        os.makedirs("models", exist_ok=True)
        joblib.dump(list(X.columns), "models/feature_columns.pkl")
        mlflow.log_artifact("models/feature_columns.pkl")

        # Save local copy & register in MLflow
        joblib.dump(model, "models/champion_model.pkl")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="HeartDisease_Champion"
        )
        print("✅ CHAMPION Model logged and registered in MLflow!")

if __name__ == "__main__":
    train_champion()