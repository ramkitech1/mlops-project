import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocess import load_data, preprocess

mlflow.set_experiment("mlops-project")

def train():
    df = load_data("data/data.csv")
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "models/model.pkl")

    print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train()