from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
RESULTS_DIR = BASE_DIR / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nInformações gerais:")
    print(df.info())

    print("\nValores ausentes:")
    print(df.isnull().sum())

    print("\nDistribuição das espécies:")
    print(df["species"].value_counts())

    print("\nEstatísticas descritivas:")
    print(df.describe())

    numeric_columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]

    df[numeric_columns].hist(figsize=(10, 8), bins=15)
    plt.suptitle("Distribuição das características do Iris")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribuicao_variaveis.png", dpi=150)
    plt.close()

    for species, group in df.groupby("species"):
        plt.scatter(
            group["petal_length_cm"],
            group["petal_width_cm"],
            label=species,
            alpha=0.8,
        )

    plt.xlabel("Comprimento da pétala (cm)")
    plt.ylabel("Largura da pétala (cm)")
    plt.title("Relação entre comprimento e largura da pétala")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "petalas_por_especie.png", dpi=150)
    plt.close()

    print("\nGráficos salvos na pasta results.")


if __name__ == "__main__":
    main()
