# Iris ML

Projeto de Machine Learning para classificação de espécies de flores do conjunto de dados Iris a partir de características da sépala e da pétala.

## Objetivo

O objetivo do projeto é aplicar um fluxo básico de aprendizado supervisionado, passando pela obtenção dos dados, análise exploratória, treinamento de diferentes algoritmos de classificação, avaliação dos resultados e utilização do melhor modelo para realizar novas previsões.

## Dataset

Foi utilizado o conjunto de dados Iris disponibilizado pelo `scikit-learn`.

A base possui 150 amostras divididas igualmente entre três espécies:

- `setosa`
- `versicolor`
- `virginica`

Cada amostra possui quatro características:

- comprimento da sépala;
- largura da sépala;
- comprimento da pétala;
- largura da pétala.

O arquivo `src/download_data.py` carrega a base pelo `scikit-learn`, organiza os nomes das colunas e gera o arquivo `data/iris.csv`.

## Tecnologias utilizadas

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Estrutura do projeto

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
│   ├── download_data.py
│   ├── exploration.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Funcionamento

O projeto foi dividido em quatro etapas principais.

### 1. Obtenção dos dados

Execute:

```bash
python src/download_data.py
```

O script utiliza `load_iris()` do `scikit-learn` e salva a base em:

```text
data/iris.csv
```

### 2. Análise exploratória

Execute:

```bash
python src/exploration.py
```

Nesta etapa são verificados:

- primeiras linhas da base;
- tipos das variáveis;
- valores ausentes;
- distribuição das espécies;
- estatísticas descritivas.

Também são gerados gráficos na pasta `results/`.

### 3. Treinamento e avaliação

Execute:

```bash
python src/train.py
```

Os dados são separados em 80% para treinamento e 20% para teste, mantendo a proporção entre as três espécies.

Foram comparados três algoritmos:

- K-Nearest Neighbors (KNN);
- Decision Tree;
- Random Forest.

A avaliação utiliza acurácia, precision, recall, F1-score e matriz de confusão.

Na execução utilizada neste projeto, os resultados obtidos foram:

| Modelo        | Acurácia |
| ------------- | -------: |
| KNN           |     100% |
| Decision Tree |   93,33% |
| Random Forest |      90% |

Os resultados podem variar caso sejam alterados os parâmetros ou a forma de divisão dos dados.

O modelo com melhor desempenho é salvo em:

```text
models/best_model.joblib
```

### 4. Previsão de novas amostras

Após o treinamento, execute:

```bash
python src/predict.py
```

O programa solicitará quatro medidas, em centímetros:

```text
Comprimento da sépala
Largura da sépala
Comprimento da pétala
Largura da pétala
```

Exemplo:

```text
5.1
3.5
1.4
0.2
```

Para valores com esse perfil, o modelo tende a classificar a amostra como `setosa`.

## Como executar o projeto

Clone o repositório e entre na pasta:

```bash
git clone URL_DO_REPOSITORIO
cd iris-ml
```

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

Depois execute os scripts na seguinte ordem:

```bash
python src/download_data.py
python src/exploration.py
python src/train.py
python src/predict.py
```

## Resultados gerados

A pasta `results/` contém:

- distribuição das variáveis;
- relação entre comprimento e largura das pétalas;
- comparação de desempenho dos modelos;
- matriz de confusão do modelo selecionado.

## Conceitos aplicados

Durante o desenvolvimento foram aplicados conceitos de:

- análise exploratória de dados;
- aprendizado supervisionado;
- classificação;
- divisão entre treino e teste;
- comparação de algoritmos;
- métricas de avaliação;
- persistência de modelos;
- inferência com novos dados.

## Autor

Victor Mendes
