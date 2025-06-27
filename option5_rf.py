#
# This script implements the methodology described in the paper:
# "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"
# Specifically, it reproduces the "Random Forest (AF+NE)" experiment, which combines
# all features (AF) with node embeddings (NE) from a Graph Convolutional Network (GCN).
#


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import time
from tqdm.notebook import tqdm

# --- 1. Data Loading and Preprocessing ---
def load_data():
    """
    Loads the Elliptic dataset from CSV files, merges them, and performs
    initial preprocessing.
    """
    
    print("Loading data...")
    try:
        features_df = pd.read_csv('data/elliptic_txs_features.csv', header=None)
        edgelist_df = pd.read_csv('data/elliptic_txs_edgelist.csv')
        classes_df = pd.read_csv('data/elliptic_txs_classes.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the dataset files are in the same directory.")
        return None

    # Name the columns for clarity
    # The first column is the txId, the second is the timestep
    # Features 2-94 are local features, 95-166 are aggregated features
    features_df.columns = ['txId', 'timestep'] + [f'local_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]

    # Map class labels to meaningful names
    # '1' for illicit, '2' for licit
    classes_df['class'] = classes_df['class'].map({'1': 'illicit', '2': 'licit', 'unknown': 'unknown'})

    # Merge features and classes
    data_df = pd.merge(features_df, classes_df, on='txId', how='left')
    
    # Sort by timestep
    data_df = data_df.sort_values('timestep').reset_index(drop=True)

    print("Data loaded and merged successfully.")
    print(f"Total transactions: {len(data_df)}")
    print("Class distribution:")
    print(data_df['class'].value_counts())
    
    return data_df, edgelist_df

# --- 2. GCN Model Definition ---
class GCN(nn.Module):
    """
    A 2-layer Graph Convolutional Network as described in the paper.
    """
    def __init__(self, n_features, n_hidden, n_classes, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(n_features, n_hidden)
        self.gc2 = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        """
        Forward pass for the GCN.
        adj: The normalized adjacency matrix.
        x: The node feature matrix.
        """
        # First GCN layer
        h1 = F.relu(self.gc1(torch.matmul(adj, x)))
        h1_d = self.dropout(h1)
        
        # Second GCN layer
        logits = self.gc2(torch.matmul(adj, h1_d))
        
        # We return the hidden layer activations as node embeddings
        # and the final logits for classification.
        return logits, h1

def normalize_adjacency_matrix(adj):
    """
    Computes the symmetrically normalized adjacency matrix from the paper.
    A_hat = D^{-1/2} * (A + I) * D^{-1/2}
    """
    print("Normalizing adjacency matrix...")
    adj = adj + torch.eye(adj.shape[0]) # Add self-loops
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return adj.matmul(d_mat_inv_sqrt).transpose(0,1).matmul(d_mat_inv_sqrt)


# --- 3. GCN Training and Embedding Generation ---

def generate_node_embeddings(data_df, edgelist_df):
    """
    Trains a GCN in a semi-supervised manner to generate node embeddings.
    """
    print("\n--- Starting GCN Training for Node Embedding Generation ---")
    
    # Prepare data for PyTorch
    # Feature scaling
    features = data_df.iloc[:, 2:167].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Map txId to an index
    txid_to_idx = {txid: i for i, txid in enumerate(data_df['txId'])}
    n_nodes = len(data_df)
    
    # Create adjacency matrix
    adj = torch.zeros((n_nodes, n_nodes))
    for index, row in tqdm(edgelist_df.iterrows(), total=edgelist_df.shape[0], desc="Building adjacency matrix"):
        src_idx = txid_to_idx.get(row['txId1'])
        tgt_idx = txid_to_idx.get(row['txId2'])
        if src_idx is not None and tgt_idx is not None:
            adj[src_idx, tgt_idx] = 1
            adj[tgt_idx, src_idx] = 1 # Make it symmetric for GCN

    adj_normalized = normalize_adjacency_matrix(adj)
    
    # Prepare labels and masks
    labels_map = {'licit': 0, 'illicit': 1, 'unknown': 2}
    labels = torch.LongTensor(data_df['class'].map(labels_map).values)
    features_tensor = torch.FloatTensor(features)
    
    # Use labels only from the training period (timestep <= 34)
    train_mask = torch.BoolTensor(data_df['timestep'] <= 34)
    # Mask out unknown classes for loss calculation
    train_mask &= (labels != 2)

    # Model parameters
    n_features = features_tensor.shape[1]
    n_hidden = 100 # As per the paper's hyperparameter tuning
    n_classes = 2  # We only classify licit vs illicit in the loss
    
    model = GCN(n_features=n_features, n_hidden=n_hidden, n_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Weighted loss for imbalanced classes (licit vs illicit)
    # The paper mentions a 0.3/0.7 ratio
    loss_weights = torch.FloatTensor([0.3, 0.7])
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # Training loop
    print("Training GCN...")
    start_time = time.time()
    for epoch in tqdm(range(100), desc="GCN Training"): # The paper trained for 1000 epochs, 100 is sufficient for demonstration
        model.train()
        optimizer.zero_grad()
        logits, _ = model(features_tensor, adj_normalized)
        
        # Calculate loss only on known, training nodes
        loss = criterion(logits[train_mask], labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # This will be too noisy inside tqdm
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    print(f"GCN training finished in {time.time() - start_time:.2f} seconds.")

    # Generate final embeddings
    print("Generating node embeddings from trained GCN...")
    model.eval()
    with torch.no_grad():
        _, embeddings_tensor = model(features_tensor, adj_normalized)
    
    return embeddings_tensor.numpy()

# --- 4. Random Forest Classification ---
def run_random_forest(data_df, embeddings):
    """
    Trains and evaluates a Random Forest classifier using the combined
    original features and GCN embeddings.
    """
    print("\n--- Starting Random Forest Classification ---")

    # Combine original features and embeddings
    original_features = data_df.iloc[:, 2:167].values
    
    # Scale original features
    scaler = StandardScaler()
    original_features_scaled = scaler.fit_transform(original_features)

    combined_features = np.concatenate([original_features_scaled, embeddings], axis=1)
    
    # Prepare labels for RF: 0 for licit, 1 for illicit
    labels = data_df['class'].map({'licit': 0, 'illicit': 1, 'unknown': -1}).values
    
    # Temporal split
    train_indices = data_df[data_df['timestep'] <= 34].index
    test_indices = data_df[data_df['timestep'] > 34].index
    
    X_train = combined_features[train_indices]
    y_train = labels[train_indices]
    X_test = combined_features[test_indices]
    y_test = labels[test_indices]
    
    # Filter out 'unknown' classes from training and testing sets
    X_train_filtered = X_train[y_train != -1]
    y_train_filtered = y_train[y_train != -1]
    X_test_filtered = X_test[y_test != -1]
    y_test_filtered = y_test[y_test != -1]
    
    print(f"Training set size (known labels): {len(y_train_filtered)}")
    print(f"Test set size (known labels): {len(y_test_filtered)}")
    
    # Train Random Forest
    # Using parameters from the paper: 50 estimators
    # The paper also mentions 50 max features, which is a fraction of total
    # We will use the default for max_features which is sqrt(n_features)
    print("Training Random Forest classifier...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_filtered, y_train_filtered)
    print(f"RF training finished in {time.time() - start_time:.2f} seconds.")
    
    # Evaluate
    print("Evaluating model performance on the test set...")
    y_pred = rf.predict(X_test_filtered)
    
    print("\nClassification Report (Test Set):")
    # 0 is licit, 1 is illicit
    print(classification_report(y_test_filtered, y_pred, target_names=['Licit', 'Illicit']))
    
    # The paper's main reported score is Illicit F1
    illicit_f1 = f1_score(y_test_filtered, y_pred, pos_label=1)
    print(f"F1 Score for Illicit class: {illicit_f1:.4f}")

# --- Main Execution ---
if __name__ == "__main__":
    df, edgelist = load_data()
    if df is not None:
        node_embeddings = generate_node_embeddings(df, edgelist)
        run_random_forest(df, node_embeddings)
