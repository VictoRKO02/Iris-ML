from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )

    prediction = model.predict(sample)[0]
    return prediction


def main():
    print("Classificação de uma nova flor Iris")

    sepal_length = float(input("Comprimento da sépala (cm): "))
    sepal_width = float(input("Largura da sépala (cm): "))
    petal_length = float(input("Comprimento da pétala (cm): "))
    petal_width = float(input("Largura da pétala (cm): "))

    prediction = predict_flower(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
    )

    print(f"\nEspécie prevista: {prediction}")


if __name__ == "__main__":
    main()
