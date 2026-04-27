import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def predict(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return int(prediction[0])