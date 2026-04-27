import logging
import joblib
import numpy as np
import os

#  Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

#  Configure logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

#  Load model
model = joblib.load("models/model.pkl")

def predict(data):
    data_array = np.array(data).reshape(1, -1)

    prediction = model.predict(data_array)[0]

    #  Log input + output
    logging.info(f"Input: {data} | Prediction: {prediction}")

    return int(prediction)