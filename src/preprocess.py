import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y