import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from preprocess import load_data, preprocess





#  Force same tracking location
mlflow.set_tracking_uri("file:./mlruns")

def train():
    df = load_data("data/data.csv")
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    #  IMPORTANT: create experiment FIRST
    experiment_name = "mlops-project"
    mlflow.set_experiment(experiment_name)

    #  Explicitly bind run to experiment
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):

        pipeline.fit(X_train, y_train)

        accuracy = pipeline.score(X_test, y_test)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(pipeline, name="model")

        print(f" Accuracy: {accuracy}")

if __name__ == "__main__":
    train()