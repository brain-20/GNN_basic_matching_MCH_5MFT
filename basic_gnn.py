"""
Dependencies:
  - Python 3.x
  - numpy
  - pandas
  - torch (PyTorch)
  - scikit-learn
  - matplotlib
  - torch_geometric

Basic GNN for Edge Classification

This script processes a dataset to build a heterogeneous graph:
  - Unique MCH tracks (columns 1-5) become nodes.
  - Each row (candidate match) becomes an MFT node.
  - An edge connects an MCH node with its corresponding MFT node.

The model uses two rounds of message passing (GraphSAGE via HeteroConv)
and an MLP to predict the probability that an edge is a true match.
It then plots the ROC curve and the Precision-Recall curve.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import PyTorch Geometric modules
try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not installed. GNN code will be skipped.")
    PYG_AVAILABLE = False

# Global configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters for training
EPOCHS = 2000
LEARNING_RATE = 0.005
USE_EARLY_STOPPING = False
EARLY_STOP_PATIENCE = 500
DROPOUT_RATE = 0.1

# ------------------ Data Loading and Splitting ------------------ #
def group_based_train_test_split(data, group_col='MCH_id', test_size=0.3, random_state=42):
    """
    Split the data into train and test sets by groups.
    All rows with the same group (MCH_id) are kept in either the train or test set.
    """
    unique_groups = data[group_col].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
    train_mask = data[group_col].isin(train_groups)
    train_df = data[train_mask].reset_index(drop=True)
    test_df = data[~train_mask].reset_index(drop=True)
    return train_df, test_df

def load_data_and_split(file_path, test_size=0.3, random_state=42):
    """
    Load the dataset and split it based on MCH tracks.
    The first 5 columns represent the MCH features and are used to create a unique MCH identifier.
    """
    column_names = [
        'X_MCH', 'Y_MCH', 'phi_MCH', 'tanL_MCH', 'invqpt_MCH',
        'X_MFT', 'Y_MFT', 'phi_MFT', 'tanL_MFT', 'invqpt_MFT',
        'MCHcov1', 'MCHcov2', 'MCHcov3', 'MCHcov4', 'MCHcov5',
        'MFTcov1', 'MFTcov2', 'MFTcov3', 'MFTcov4', 'MFTcov5',
        'chi2', 'Match', 'IsMatchable', 'pT'
    ]
    data = pd.read_csv(file_path, sep=" ", header=None, names=column_names)

    # Generate a unique identifier for each MCH track using the first 5 columns.
    mch_cols = ['X_MCH', 'Y_MCH', 'phi_MCH', 'tanL_MCH', 'invqpt_MCH']
    data['MCH_id'] = data[mch_cols].apply(lambda row: tuple(row), axis=1).astype('category').cat.codes

    train_df, test_df = group_based_train_test_split(data, group_col='MCH_id', test_size=test_size, random_state=random_state)
    return train_df, test_df

# ------------------ Graph Construction for the GNN ------------------ #
def build_heterodata_for_basic_gnn(train_df, test_df, use_scaling=True):
    """
    Build a heterogeneous graph:
      - Unique MCH tracks (columns 1-5) form the 'MCH' nodes.
      - Each row becomes an 'MFT' node.
      - An edge connects an MCH node to its associated MFT node.
      - Additional edge features are calculated.
    """
    train_df = train_df.copy()
    train_df['split'] = 'train'
    test_df = test_df.copy()
    test_df['split'] = 'test'
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    mch_cols = ['X_MCH', 'Y_MCH', 'phi_MCH', 'tanL_MCH', 'invqpt_MCH']
    unique_mch = full_df[mch_cols].drop_duplicates().reset_index(drop=True)
    mch_to_id = {tuple(row): idx for idx, row in unique_mch.iterrows()}
    full_df['MCH_id_final'] = full_df[mch_cols].apply(lambda r: mch_to_id[tuple(r)], axis=1)

    # Process MFT features (each row is a candidate node)
    mft_cols = ['X_MFT', 'Y_MFT', 'phi_MFT', 'tanL_MFT', 'invqpt_MFT']
    mft_features = full_df[mft_cols].values.astype(np.float32)
    if use_scaling:
        scaler_mft = StandardScaler()
        mft_features = scaler_mft.fit_transform(mft_features)
    
    # Process MCH node features
    mch_features = unique_mch.values.astype(np.float32)
    if use_scaling:
        scaler_mch = StandardScaler()
        mch_features = scaler_mch.fit_transform(mch_features)

    # Create the heterogeneous graph object
    data_hetero = HeteroData()
    data_hetero['MCH'].x = torch.tensor(mch_features, dtype=torch.float32).to(device)
    data_hetero['MFT'].x = torch.tensor(mft_features, dtype=torch.float32).to(device)

    # Create edges between MCH and MFT nodes
    edge_src = full_df['MCH_id_final'].values
    edge_dst = np.arange(full_df.shape[0])
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
    data_hetero['MCH', 'match', 'MFT'].edge_index = edge_index
    data_hetero['MFT', 'rev_match', 'MCH'].edge_index = edge_index[[1, 0]]

    # Attach binary edge labels
    edge_label = full_df['Match'].values.astype(np.float32)
    data_hetero['MCH', 'match', 'MFT'].edge_label = torch.tensor(edge_label, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Compute additional edge features
    chi2 = full_df['chi2'].values.reshape(-1, 1).astype(np.float32)
    diff = mft_features - mch_features[edge_src]
    norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([chi2, diff, norm_diff], axis=1)
    data_hetero['MCH', 'match', 'MFT'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(device)

    # Create train/test masks for edges
    full_splits = full_df['split'].values
    train_mask = (full_splits == 'train')
    test_mask = (full_splits == 'test')
    data_hetero['MCH', 'match', 'MFT'].train_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
    data_hetero['MCH', 'match', 'MFT'].test_mask = torch.tensor(test_mask, dtype=torch.bool).to(device)

    return data_hetero

# ------------------ GNN Model Definition ------------------ #
class BasicGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, dropout_rate=0.1):
        """
        Initialize the model:
          - Raw node features are projected into a latent space.
          - Two rounds of heterogeneous message passing update node embeddings.
          - For each edge, concatenate the embeddings from source and destination.
          - An MLP predicts the probability of a true match.
        """
        super().__init__()
        self.lin_mch = nn.Linear(input_dim, hidden_dim)
        self.lin_mft = nn.Linear(input_dim, hidden_dim)
        
        self.conv1 = HeteroConv({
            ('MCH', 'match', 'MFT'): SAGEConv(hidden_dim, hidden_dim),
            ('MFT', 'rev_match', 'MCH'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('MCH', 'match', 'MFT'): SAGEConv(hidden_dim, hidden_dim),
            ('MFT', 'rev_match', 'MCH'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='sum')
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, data):
        """
        Forward pass:
          1. Transform node features and apply dropout.
          2. Apply two rounds of message passing.
          3. For each edge, concatenate the embeddings from source (MCH) and destination (MFT).
          4. Classify the edge using an MLP.
        """
        x_dict = {
            'MCH': self.dropout(self.lin_mch(data['MCH'].x)),
            'MFT': self.dropout(self.lin_mft(data['MFT'].x))
        }
        
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        src, dst = data['MCH', 'match', 'MFT'].edge_index
        edge_input = torch.cat([x_dict['MCH'][src], x_dict['MFT'][dst]], dim=1)
        return self.edge_classifier(edge_input)

# ------------------ Training and Evaluation ------------------ #
def train_and_eval_basic_gnn(data_hetero):
    """
    Train the BasicGNN model and evaluate on test edges.
    Returns a dictionary of evaluation metrics and the trained model.
    """
    model = BasicGNN(input_dim=5, hidden_dim=64, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_mask = data_hetero['MCH', 'match', 'MFT'].train_mask
    test_mask = data_hetero['MCH', 'match', 'MFT'].test_mask
    edge_label = data_hetero['MCH', 'match', 'MFT'].edge_label
    
    best_test_auc = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        preds = model(data_hetero)
        loss = criterion(preds[train_mask], edge_label[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = preds[test_mask].cpu().numpy()
                test_true = edge_label[test_mask].cpu().numpy()
                test_auc = roc_auc_score(test_true, test_preds)
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if USE_EARLY_STOPPING and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}, best test AUC: {best_test_auc:.3f}")
                break

    model.eval()
    with torch.no_grad():
        all_preds = model(data_hetero).cpu().numpy()
    all_true = edge_label.cpu().numpy().ravel()
    
    train_mask_np = data_hetero['MCH', 'match', 'MFT'].train_mask.cpu().numpy()
    test_mask_np = data_hetero['MCH', 'match', 'MFT'].test_mask.cpu().numpy()
    
    train_probs = all_preds[train_mask_np].ravel()
    test_probs = all_preds[test_mask_np].ravel()
    
    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)
    
    train_auc = roc_auc_score(all_true[train_mask_np], train_probs)
    test_auc = roc_auc_score(all_true[test_mask_np], test_probs)
    train_f1 = f1_score(all_true[train_mask_np], train_preds)
    test_f1 = f1_score(all_true[test_mask_np], test_preds)
    test_cm = confusion_matrix(all_true[test_mask_np], test_preds)
    
    s = test_cm[1, 1]
    b = test_cm[0, 1]
    test_sbr = s / (s + b) if (s + b) > 0 else 0
    test_recall = s / (s + test_cm[1, 0]) if (s + test_cm[1, 0]) > 0 else 0
    test_fp_rate = b / (b + test_cm[0, 0]) if (b + test_cm[0, 0]) > 0 else 0
    fpr, tpr, _ = roc_curve(all_true, all_preds)
    precision, recall_vals, _ = precision_recall_curve(all_true, all_preds)
    pr_auc = average_precision_score(all_true, all_preds)
    
    metrics = {
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_cm': test_cm,
        'test_sbr': test_sbr,
        'test_recall': test_recall,
        'test_fp_rate': test_fp_rate,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall_vals
    }
    
    return metrics, model

# ------------------ Plotting Functions ------------------ #
def plot_roc_curve(fpr, tpr, auc_score):
    """Plot the ROC Curve."""
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(precision, recall, pr_auc):
    """Plot the Precision-Recall Curve."""
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.show()

# ------------------ Main Function ------------------ #
def main():
    # Set the path to your dataset file.
    file_path = r"C:\Users\salma\Desktop\Masters\Machine_Learning\AO2D_5ao5_5cand.txt"
    
    # Load and split the data.
    train_df, test_df = load_data_and_split(file_path, test_size=0.3, random_state=42)
    
    # Build the heterogeneous graph.
    data_gnn = build_heterodata_for_basic_gnn(train_df, test_df, use_scaling=True)
    
    # Train the model and obtain evaluation metrics.
    print("\n=== Training GNN ===")
    metrics, model = train_and_eval_basic_gnn(data_gnn)
    
    # Print evaluation metrics.
    print("\nGNN Evaluation Metrics:")
    for key, value in metrics.items():
        if key in ['fpr', 'tpr', 'precision', 'recall']:
            continue
        print(f"{key}: {value}")
    
    # Plot the ROC and Precision-Recall curves.
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['test_auc'])
    plot_precision_recall_curve(metrics['precision'], metrics['recall'], metrics['pr_auc'])
    
if __name__ == "__main__":
    if PYG_AVAILABLE:
        main()
    else:
        print("PyTorch Geometric is required to run the GNN code.")
