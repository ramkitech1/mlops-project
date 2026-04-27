from fastapi import FastAPI
from src.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MLOps API Running"}

@app.post("/predict")
def make_prediction(age: int, salary: int, experience: int):
    result = predict([age, salary, experience])
    return {"prediction": result}