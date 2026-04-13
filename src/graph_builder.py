# ==============================================================================
# Graph Builder — Global Graph + Ego Graph Construction
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
        x = np.zeros((self.total_nodes, 12), dtype=np.float32)

        # ---- xApp nodes ----
        xapp_data = df_t.drop_duplicates('xapp_id').set_index('xapp_id')
        for xid in range(self.num_xapps):
            idx     = xid
            x[idx, 0] = 1.0  # is_xapp
            if xid in xapp_data.index:
                feats    = xapp_data.loc[xid, ['cpu_usage', 'memory_usage', 'rmr_msg_rate', 'prb_usage_dl']].values.reshape(1, -1)
                x[idx, 3:7] = self.scaler_xapp.transform(feats)

        # ---- Cell nodes ----
        cell_data = df_t.drop_duplicates('target_cell').set_index('target_cell')
        for cid in range(self.num_cells):
            idx     = self.num_xapps + cid
            x[idx, 1] = 1.0  # is_cell
            if cid in cell_data.index:
                feats    = cell_data.loc[cid, ['cell_load', 'ue_count']].values.reshape(1, -1)
                x[idx, 7:9] = self.scaler_cell.transform(feats)

        # ---- UE nodes ----
        ue_data = df_t.drop_duplicates('target_ue').set_index('target_ue')
        for uid in range(self.num_ues):
            idx     = self.num_xapps + self.num_cells + uid
            x[idx, 2] = 1.0  # is_ue
            if uid in ue_data.index:
                feats    = ue_data.loc[uid, ['rsrp', 'sinr', 'throughput']].values.reshape(1, -1)
                x[idx, 9:12] = self.scaler_ue.transform(feats)

        # ---- Edges ----
        src, dst = [], []

        # xApp <-> UE (deduplicated)
        pairs = df_t[['xapp_id', 'target_ue']].drop_duplicates()
        for _, row in pairs.iterrows():
            u = int(row['xapp_id'])
            v = self.num_xapps + self.num_cells + int(row['target_ue'])
            src += [u, v]; dst += [v, u]

        # UE <-> Cell
        for uid, row in ue_data.iterrows():
            u = self.num_xapps + self.num_cells + int(uid)
            c = self.num_xapps + int(row['target_cell'])
            src += [u, c]; dst += [c, u]

        # ---- Node-level labels ----
        y = torch.full((self.total_nodes,), -1, dtype=torch.long)
        for xid in range(self.num_xapps):
            if xid in xapp_data.index:
                y[xid] = int(xapp_data.loc[xid, 'is_malicious'])
            else:
                y[xid] = 0

        return Data(
            x          = torch.tensor(x, dtype=torch.float),
            edge_index = torch.tensor([src, dst], dtype=torch.long),
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
