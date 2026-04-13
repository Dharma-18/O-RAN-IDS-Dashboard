# ==============================================================================
# Model Definition — AttentionGuidedEgoGAT + NT-Xent Loss
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GraphNorm
from torch_geometric.utils import dropout_edge

# ==============================================================================
# CONTRASTIVE LOSS
# ==============================================================================
def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss for Graph Contrastive Learning."""
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
    def __init__(self, num_node_features=12, hidden_channels=64, heads=4, num_classes=1, dropout=0.5, edge_dropout=0.2):
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

    def forward(self, data, return_embeds=False, capture_attention=False):
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
        
        # Streamlit App dependency handling
        if capture_attention:
            x2, (attn_ei, attn_w) = self.conv2(x, edge_index, return_attention_weights=True)
        else:
            x2 = self.conv2(x, edge_index)
            attn_ei, attn_w = None, None
            
        x2 = self.norm2(x2, batch)
        x = F.gelu(x2 + res2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if hasattr(data, 'root_node') and data.root_node is not None:
            ptr = data.ptr
            # Convert scalar tensor root_node robustly
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

        if capture_attention:
            return logits, embeds, attn_ei, attn_w
        if return_embeds:
            return logits, embeds
        return logits
