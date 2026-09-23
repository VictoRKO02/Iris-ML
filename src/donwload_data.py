from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "iris.csv"

iris = load_iris(as_frame=True)

df = iris.frame.copy()

df = df.rename(columns={
    "sepal length (cm)": "sepal_length_cm",
    "sepal width (cm)": "sepal_width_cm",
    "petal length (cm)": "petal_length_cm",
    "petal width (cm)": "petal_width_cm",
    "target": "target",
})

df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica",
})

DATA_DIR.mkdir(exist_ok=True)

df.to_csv(DATA_PATH, index=False)

print(f"Dataset salvo em: {DATA_PATH}")