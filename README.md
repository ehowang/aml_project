# Illicit Bitcoin Transaction Detection

This project focuses on identifying illicit transactions in the Bitcoin network using the Elliptic dataset. The primary goal is to build and evaluate various machine learning models to classify transactions as 'licit' or 'illicit'.

## Dataset

The project utilizes the [Elliptic Dataset](https://www.kaggle.com/datasets/elliptic-data-set/elliptic-data-set), which is a graph dataset of Bitcoin transactions. It contains:
- `elliptic_txs_classes.csv`: A file that maps transaction IDs to classes ('unknown', '1' for illicit, '2' for licit).
- `elliptic_txs_edgelist.csv`: An edge list representing the transaction graph.
- `elliptic_txs_features.csv`: A file containing transaction features. These are anonymized and include local features and aggregated features over the neighborhood of transactions in the graph.

## Project Structure

- **`data/`**: Contains the Elliptic dataset files.
- **`*.ipynb`**: Jupyter notebooks with different experiments and models. These include:
    - `eda.ipynb`: The main notebook for data exploration and feature engineering.
    - `LogisticRegression.ipynb`: Implements a Logistic Regression model.
    - `Option5_GCN.ipynb`: Implements a Graph Convolutional Network (GCN) model.
    - `Option5_RF_NE.ipynb`: Implements a Random Forest model with Node Embeddings.
    - `baseline_calculation.ipynb`: Active Learning: IF + Uncertainty Sampling + Logistic Regression
    - `Option1-Option4`: adopt different active learning methods
- **`results.json`**: Stores results from model evaluations.
- **`seeds.txt`**: Contains random seeds for reproducibility.

## Setup

To set up the environment and run the notebooks, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd aml_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the code using Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab
```

Then, open the desired notebook (e.g., `Grp Proj.ipynb`) and run the cells.
