# Iris ML

Projeto de classificação de espécies de flores utilizando técnicas de Machine Learning.

O objetivo é analisar o conjunto de dados Iris e construir modelos capazes de identificar a espécie de uma flor a partir de medidas da sépala e da pétala.

## Dataset

O projeto utiliza o Iris Dataset, composto por 150 amostras distribuídas entre três espécies:

- Iris setosa
- Iris versicolor
- Iris virginica

Cada registro possui quatro atributos:

- comprimento da sépala
- largura da sépala
- comprimento da pétala
- largura da pétala

Fonte original: UCI Machine Learning Repository.

## Tecnologias

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Estrutura

```text
iris-ml/
├── data/
│   └── iris.csv
├── models/
│   └── best_model.joblib
├── results/
│   ├── confusion_matrix.png
│   ├── distribuicao_variaveis.png
│   ├── model_comparison.csv
│   └── petalas_por_especie.png
├── src/
│   ├── exploration.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Etapas do projeto

1. Leitura e inspeção do conjunto de dados.
2. Verificação de valores ausentes e distribuição das classes.
3. Análise exploratória das variáveis.
4. Separação dos dados em treino e teste.
5. Treinamento de três algoritmos de classificação.
6. Comparação das acurácias.
7. Seleção e armazenamento do melhor modelo.
8. Classificação de novas amostras.

## Modelos utilizados

Foram comparados três algoritmos:

- K-Nearest Neighbors
- Decision Tree
- Random Forest

O conjunto de dados é dividido em 80% para treino e 20% para teste, utilizando divisão estratificada para manter a proporção das espécies.

## Como executar

Crie um ambiente virtual:

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a análise exploratória:

```bash
python src/exploration.py
```

Treine e compare os modelos:

```bash
python src/train.py
```

Depois do treinamento, execute uma previsão:

```bash
python src/predict.py
```

Informe as quatro medidas solicitadas pelo programa.

## Exemplo

Uma amostra com valores:

```text
Comprimento da sépala: 5.1
Largura da sépala: 3.5
Comprimento da pétala: 1.4
Largura da pétala: 0.2
```

deve ser classificada como uma flor próxima ao padrão da espécie setosa.

## Objetivo acadêmico

O projeto foi desenvolvido para aplicar conceitos fundamentais de aprendizado supervisionado, incluindo preparação de dados, classificação, comparação de algoritmos, avaliação de desempenho e uso de um modelo treinado para realizar novas previsões.
