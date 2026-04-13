import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GraphNorm
from torch_geometric.utils import dropout_edge
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    precision_recall_curve, confusion_matrix, roc_curve, auc
)

# --- Imports needed to generate ego_dataset ---
from src.des_generator import DatasetGenerator, DEFAULT_CONFIG
from src.graph_builder import GlobalGraphBuilder, generate_ego_graphs

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TRAIN_SPLITS_TO_TEST = [0.6] 
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 100             
SEED = 42
CONTRASTIVE_WEIGHT = 0.1  
CONTRASTIVE_TEMP = 0.5    

torch.manual_seed(SEED)
np.random.seed(SEED)

# Global list to store experiment results
all_experiment_results = []
epoch_logs = []

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================
def get_optimal_threshold(labels, logits):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1_scores = np.divide(
        2 * precisions * recalls, 
        precisions + recalls, 
        out=np.zeros_like(precisions), 
        where=(precisions + recalls) != 0
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold

def nt_xent_loss(z1, z2, temperature=0.5):
    device = z1.device
    batch_size = z1.size(0)
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    targets = torch.cat([torch.arange(batch_size, 2*batch_size), 
                         torch.arange(0, batch_size)]).to(device)
    return F.cross_entropy(sim_matrix, targets)

# ==============================================================================
# 3. MODEL DEFINITION
# ==============================================================================
class AttentionGuidedEgoGAT(nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=4, num_classes=1, dropout=0.5, edge_dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.ego_gate = nn.Linear(hidden_channels, hidden_channels * 2)

        self.res_proj1 = nn.Linear(num_node_features, hidden_channels * heads)
        self.res_proj2 = nn.Linear(hidden_channels * heads, hidden_channels)

        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=heads, dropout=dropout)
        self.norm1 = GraphNorm(hidden_channels * heads)

        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.norm2 = GraphNorm(hidden_channels)

        self.fc1 = nn.Linear(hidden_channels * 3, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data, return_embeds=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)

        res1 = self.res_proj1(x)
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1, batch)
        x = F.gelu(x1 + res1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        res2 = self.res_proj2(x)
        x2 = self.conv2(x, edge_index)
        x2 = self.norm2(x2, batch)
        x = F.gelu(x2 + res2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if hasattr(data, 'root_node') and data.root_node is not None:
            ptr = data.ptr
            # Safe logic for handling scalar root tensors vs batched roots
            if len(ptr) > 1 and not isinstance(data.root_node, list) and data.root_node.dim() == 1:
                root_indices = ptr[:-1] + data.root_node
            else:
                 r_nodes = []
                 for i in range(len(ptr) - 1):
                     r_node = data.root_node[i].item() if isinstance(data.root_node, torch.Tensor) else data.root_node[i]
                     r_nodes.append(ptr[i].item() + r_node)
                 root_indices = torch.tensor(r_nodes, device=x.device)

            x_root = x[root_indices]
        else:
            x_root = global_mean_pool(x, batch)

        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_neigh = torch.cat([x_mean, x_max], dim=1)

        gate = torch.sigmoid(self.ego_gate(x_root))
        x_neigh = gate * x_neigh
        x_final = torch.cat([x_root, x_neigh], dim=1)

        embeds = F.gelu(self.fc1(x_final))
        x_drop = F.dropout(embeds, p=0.5, training=self.training)
        logits = self.fc2(x_drop)

        if return_embeds:
            return logits, embeds
        return logits

# ==============================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        target = batch.y.view(-1, 1).float()

        out1, embed1 = model(batch, return_embeds=True)
        out2, embed2 = model(batch, return_embeds=True)
        
        loss_bce = (criterion(out1, target) + criterion(out2, target)) / 2.0
        loss_con = nt_xent_loss(embed1, embed2, temperature=CONTRASTIVE_TEMP)
        loss = loss_bce + (CONTRASTIVE_WEIGHT * loss_con)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        all_logits.extend(out1.detach().cpu().numpy())
        all_labels.extend(target.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    opt_thresh = get_optimal_threshold(all_labels, all_logits)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= opt_thresh).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    
    return avg_loss, acc, f1, opt_thresh

def evaluate_epoch(model, loader, criterion, threshold, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1, 1).float()
            out = model(batch)
            loss = criterion(out, target)
            total_loss += loss.item()
            all_logits.extend(out.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {
        'loss': avg_loss,
        'acc': accuracy_score(all_labels, preds),
        'prec': precision_score(all_labels, preds, average='macro', zero_division=0),
        'rec': recall_score(all_labels, preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, preds, average='macro', zero_division=0),
        'y_true': all_labels,
        'y_probs': probs,
        'y_preds': preds
    }
    return metrics

def plot_training_results(history, phase_name, split_val):
    # Save instead of show to prevent blocking in headless runs
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['test_loss'], label='Test Loss', color='orange')
    axes[0].set_title(f'{phase_name} - Loss (Split {split_val})')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_f1'], label='Train F1', color='blue', linestyle='--')
    axes[1].plot(history['test_f1'], label='Test F1', color='orange', linestyle='--')
    axes[1].set_title(f'{phase_name} - F1 Score (Split {split_val})')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{phase_name.replace(':', '').replace(' ', '_')}_training.png")
    plt.close()

# ==============================================================================
# 5. MAIN EXPERIMENT RUNNER
# ==============================================================================
def run_experiment(train_split, ego_dataset):
    print(f"\n{'='*110}")
    print(f" EXPERIMENT: TRAIN/TEST SPLIT {train_split*100:.0f}% / {(1-train_split)*100:.0f}% ")
    print(f"{'='*110}")

    total_samples = len(ego_dataset)
    train_size = int(total_samples * train_split)
    train_dataset = ego_dataset[:train_size]
    test_dataset = ego_dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_feats = ego_dataset[0].x.shape[1] # Robust dimension check

    phases = [
        {"name": "Phase 1: Standard BCE", "weighted": False, "tag": "std"},
        {"name": "Phase 2: Weighted BCE", "weighted": True, "tag": "weighted"}
    ]

    for phase in phases:
        print(f"\n>>> STARTING {phase['name']} ...")
        
        pos_weight = None
        if phase['weighted']:
            labels = [int(d.y.item()) for d in train_dataset]
            pos_count = labels.count(1)
            neg_count = labels.count(0)
            weight_val = neg_count / pos_count if pos_count > 0 else 1.0
            pos_weight = torch.tensor([weight_val], dtype=torch.float).to(device)
            print(f"Calculated pos_weight: {weight_val:.4f} (Pos: {pos_count}, Neg: {neg_count})")

        model = AttentionGuidedEgoGAT(num_node_features=node_feats, hidden_channels=32).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        history = {'train_loss': [], 'test_loss': [], 'train_f1': [], 'test_f1': [], 'last_eval': None}

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc, tr_f1, opt_thresh = train_epoch(model, train_loader, optimizer, criterion, device)
            ts_metrics = evaluate_epoch(model, test_loader, criterion, opt_thresh, device)

            history['train_loss'].append(tr_loss)
            history['test_loss'].append(ts_metrics['loss'])
            history['train_f1'].append(tr_f1)
            history['test_f1'].append(ts_metrics['f1'])
            history['last_eval'] = ts_metrics

            epoch_logs.append({
                'Split': f"{int(train_split*100)}/{int((1-train_split)*100)}",
                'Phase': phase['name'],
                'Epoch': epoch,
                'Train_Loss': tr_loss,
                'Train_F1': tr_f1,
                'Test_Loss': ts_metrics['loss'],
                'Test_Accuracy': ts_metrics['acc'],
                'Test_Precision': ts_metrics['prec'],
                'Test_Recall': ts_metrics['rec'],
                'Test_F1': ts_metrics['f1'],
                'Threshold': opt_thresh
            })

        # Final Results Recording
        all_experiment_results.append({
            'Split': f"{int(train_split*100)}/{int((1-train_split)*100)}",
            'Phase': phase['name'],
            'Final_Test_Acc': ts_metrics['acc'],
            'Final_Test_Prec': ts_metrics['prec'],
            'Final_Test_Rec': ts_metrics['rec'],
            'Final_Test_F1': ts_metrics['f1'],
            'Final_Test_Loss': ts_metrics['loss'],
            'Optimal_Threshold': opt_thresh
        })

        model_filename = f"model_split{int(train_split*100)}_{phase['tag']}.pkl"
        torch.save({
            'model_state_dict': model.state_dict(),
            'split': train_split,
            'phase': phase['name'],
            'final_test_f1': ts_metrics['f1'],
            'opt_thresh': opt_thresh
        }, model_filename)
        print(f"\nModel saved: {model_filename}")

        plot_training_results(history, phase['name'], train_split)

# ==============================================================================
# 6. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Generating simulation data for standalone training...")
    config = DEFAULT_CONFIG.copy()
    config["topology"]["num_xapps"] = 30 # slightly larger scale for good training data
    gen = DatasetGenerator(config)
    df = gen.generate()
    builder = GlobalGraphBuilder(config)
    builder.fit(df)
    global_graphs = builder.build_window_graphs(df)
    ego_dataset = generate_ego_graphs(global_graphs, num_xapps=config["topology"]["num_xapps"])
    print(f"Ego dataset loaded with {len(ego_dataset)} samples.")

    for split in TRAIN_SPLITS_TO_TEST:
        run_experiment(split, ego_dataset)
    
    results_df = pd.DataFrame(all_experiment_results)
    results_df.to_csv("experiment_summary_results.csv", index=False)
    
    epoch_df = pd.DataFrame(epoch_logs)
    epoch_df.to_csv("epoch_wise_results.csv", index=False)
    print("\nSaved epoch-wise results to 'epoch_wise_results.csv'")
    
    print("\n" + "="*50)
    print(" FINAL SUMMARY OF ALL EXPERIMENTS ")
    print("="*50)
    print(results_df.to_string(index=False))
    print("\nFull results saved to 'experiment_summary_results.csv'")
