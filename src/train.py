from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["species"]
    return X, y


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=4,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "model": name,
            "accuracy": accuracy,
        })
        trained_models[name] = model

        print(f"\n{name}")
        print(f"Acurácia: {accuracy:.4f}")
        print(classification_report(y_test, predictions))

    results_df = pd.DataFrame(results).sort_values(
        by="accuracy",
        ascending=False,
    )
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    best_predictions = best_model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        cmap="Blues",
    )
    plt.title(f"Matriz de confusão - {best_model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nComparação dos modelos:")
    print(results_df.to_string(index=False))
    print(f"\nMelhor modelo: {best_model_name}")
    print("Modelo salvo em models/best_model.joblib")


if __name__ == "__main__":
    main()
