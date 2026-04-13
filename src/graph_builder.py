# ==============================================================================
# Graph Builder — Global Graph + Ego Graph Construction (Vectorized)
# ==============================================================================
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from sklearn.preprocessing import StandardScaler
from typing import List, Dict


class GlobalGraphBuilder:
    """
    Builds one PyG Data object (global graph) per timestamp from the DES DataFrame.
    Node types:  [0..X-1] = xApps, [X..X+C-1] = Cells, [X+C..X+C+U-1] = UEs
    Node features (12-dim):
        0   : is_xapp  (one-hot type)
        1   : is_cell
        2   : is_ue
        3-6 : xApp features  [cpu_usage, memory_usage, rmr_msg_rate, prb_usage_dl]
        7-8 : Cell features  [cell_load, ue_count]
        9-11: UE features    [rsrp, sinr, throughput]
    """

    def __init__(self, config: Dict):
        topo           = config["topology"]
        self.num_xapps = topo["num_xapps"]
        self.num_cells = topo["num_cells"]
        self.num_ues   = topo["num_ues"]
        self.total_nodes = self.num_xapps + self.num_cells + self.num_ues

        self.scaler_xapp = StandardScaler()
        self.scaler_cell = StandardScaler()
        self.scaler_ue   = StandardScaler()
        self.is_fitted   = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """Fit scalers on the full dataset before building window graphs."""
        self.scaler_xapp.fit(df[['cpu_usage', 'memory_usage', 'rmr_msg_rate', 'prb_usage_dl']].values)
        self.scaler_ue.fit(df[['rsrp', 'sinr', 'throughput']].values)
        self.scaler_cell.fit(df[['cell_load', 'ue_count']].values)
        self.is_fitted = True

    # ------------------------------------------------------------------
    def build_window_graphs(self, df_window: pd.DataFrame) -> List[Data]:
        """
        Build one global graph per unique timestamp in df_window.
        Returns a list of PyG Data objects (one per timestamp).
        """
        if not self.is_fitted:
            raise ValueError("Call .fit(df) before building graphs.")

        graphs     = []
        timestamps = sorted(df_window['timestamp'].unique())

        for t in timestamps:
            df_t = df_window[df_window['timestamp'] == t]
            graphs.append(self._build_single(df_t, t))

        return graphs

    # ------------------------------------------------------------------
    def _build_single(self, df_t: pd.DataFrame, timestamp: int) -> Data:
        X, C, U = self.num_xapps, self.num_cells, self.num_ues
        x = np.zeros((self.total_nodes, 12), dtype=np.float32)

        # ---- One-hot type encoding (vectorized) ----
        x[:X, 0] = 1.0          # xApps
        x[X:X+C, 1] = 1.0       # Cells
        x[X+C:X+C+U, 2] = 1.0   # UEs

        # ---- xApp features (vectorized) ----
        xapp_data = df_t.drop_duplicates('xapp_id').set_index('xapp_id')
        present_xapps = xapp_data.index.intersection(range(X))
        if len(present_xapps) > 0:
            feats = xapp_data.loc[present_xapps, ['cpu_usage', 'memory_usage', 'rmr_msg_rate', 'prb_usage_dl']].values
            x[present_xapps, 3:7] = self.scaler_xapp.transform(feats)

        # ---- Cell features (vectorized) ----
        cell_data = df_t.drop_duplicates('target_cell').set_index('target_cell')
        present_cells = cell_data.index.intersection(range(C))
        if len(present_cells) > 0:
            feats = cell_data.loc[present_cells, ['cell_load', 'ue_count']].values
            cell_indices = np.array(present_cells) + X
            x[cell_indices, 7:9] = self.scaler_cell.transform(feats)

        # ---- UE features (vectorized) ----
        ue_data = df_t.drop_duplicates('target_ue').set_index('target_ue')
        present_ues = ue_data.index.intersection(range(U))
        if len(present_ues) > 0:
            feats = ue_data.loc[present_ues, ['rsrp', 'sinr', 'throughput']].values
            ue_indices = np.array(present_ues) + X + C
            x[ue_indices, 9:12] = self.scaler_ue.transform(feats)

        # ---- Edges (vectorized) ----
        # xApp <-> UE
        pairs = df_t[['xapp_id', 'target_ue']].drop_duplicates()
        u_xapp = pairs['xapp_id'].values.astype(int)
        v_ue   = (X + C + pairs['target_ue'].values).astype(int)
        
        # UE <-> Cell
        if len(present_ues) > 0:
            ue_ids_arr   = np.array(present_ues, dtype=int)
            cell_ids_arr = ue_data.loc[present_ues, 'target_cell'].values.astype(int)
            u_ue_cell    = X + C + ue_ids_arr
            v_cell       = X + cell_ids_arr
        else:
            u_ue_cell = np.array([], dtype=int)
            v_cell    = np.array([], dtype=int)

        # Stack all edges (bidirectional)
        src = np.concatenate([u_xapp, v_ue,  u_ue_cell, v_cell])
        dst = np.concatenate([v_ue,  u_xapp, v_cell,    u_ue_cell])

        # ---- Node-level labels (vectorized) ----
        y = torch.full((self.total_nodes,), -1, dtype=torch.long)
        y[:X] = 0  # default benign
        if len(present_xapps) > 0:
            mal_series = xapp_data.loc[present_xapps, 'is_malicious']
            y[np.array(present_xapps)] = torch.tensor(mal_series.values.astype(int), dtype=torch.long)

        return Data(
            x          = torch.tensor(x, dtype=torch.float),
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long) if len(src) > 0 else torch.zeros((2, 0), dtype=torch.long),
            y          = y,
            timestamp  = timestamp,
        )


# ==============================================================================
# EGO GRAPH EXTRACTION
# ==============================================================================
def generate_ego_graphs(global_graphs: List[Data], num_xapps: int, num_hops: int = 2) -> List[Data]:
    """
    For every global graph and every xApp, extract a k-hop ego subgraph
    centered on that xApp.

    Returns: list of (len(global_graphs) × num_xapps) Data objects.
    """
    ego_dataset = []

    for global_data in global_graphs:
        for xid in range(num_xapps):
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx      = xid,
                num_hops      = num_hops,
                edge_index    = global_data.edge_index,
                relabel_nodes = True,
                num_nodes     = global_data.x.size(0),
            )

            ego_data = Data(
                x                = global_data.x[subset],
                edge_index       = sub_edge_index,
                y                = global_data.y[xid].unsqueeze(0),
                root_node        = mapping[0].item(),
                global_timestamp = global_data.timestamp,
                global_xapp_id   = xid,
            )
            ego_dataset.append(ego_data)

    return ego_dataset
