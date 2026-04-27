import os
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from preprocess import load_data, preprocess


def train():
    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)

    # Load & preprocess data
    df = load_data("../data/data.csv")
    X, y = preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Set MLflow tracking (local)
    mlflow.set_tracking_uri("file:../mlruns")

    # Set experiment
    experiment_name = "mlops-project"
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run():

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate
        accuracy = pipeline.score(X_test, y_test)

        # Log params & metrics
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Save model locally
        joblib.dump(pipeline, "models/model.pkl")

        print(f" Accuracy: {accuracy}")


if __name__ == "__main__":
    train()