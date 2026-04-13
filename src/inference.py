# ==============================================================================
# Inference — Batch Predictions + Attention-Weight Extraction
# ==============================================================================
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Dict, List, Any

from .model import AttentionGuidedEgoGAT


# ==============================================================================
# FULL INFERENCE
# ==============================================================================
def run_inference(
    model:        AttentionGuidedEgoGAT,
    ego_dataset:  List[Data],
    device:       torch.device,
    batch_size:   int = 32,
    threshold:    float = 0.5,
) -> Dict[str, Any]:
    """
    Two-pass inference:
      Pass 1 — Fast batched inference → predictions for every ego graph.
      Pass 2 — Individual inference with attention capture for the most
                suspicious ego graph per xApp (highest malicious probability).

    Returns a results dict with:
      all_preds     : list of per-graph dicts (xapp_id, timestamp, prob, pred, y_true)
      xapp_stats    : dict xapp_id → aggregated stats
      attention_data: dict xapp_id → detailed attention breakdown
    """
    model.eval()
    model.to(device)

    # ----------------------------------------------------------------
    # Pass 1: Batched prediction
    # ----------------------------------------------------------------
    loader    = DataLoader(ego_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch).squeeze(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > threshold).astype(int)
            y_true = batch.y.cpu().numpy()
            # global_xapp_id & global_timestamp are plain Python lists in a Batch
            xids   = np.array(batch.global_xapp_id)
            tss    = np.array(batch.global_timestamp)

            for i in range(len(probs)):
                all_preds.append({
                    'xapp_id':   int(xids[i]),
                    'timestamp': int(tss[i]),
                    'prob':      float(probs[i]),
                    'pred':      int(preds[i]),
                    'y_true':    int(y_true[i]),
                })

    # ----------------------------------------------------------------
    # Aggregate per xApp
    # ----------------------------------------------------------------
    xapp_groups: Dict[int, List] = defaultdict(list)
    for rec in all_preds:
        xapp_groups[rec['xapp_id']].append(rec)

    xapp_stats: Dict[int, Dict] = {}
    for xid, records in xapp_groups.items():
        probs_arr = np.array([r['prob']   for r in records])
        preds_arr = np.array([r['pred']   for r in records])
        true_arr  = np.array([r['y_true'] for r in records])
        xapp_stats[xid] = {
            'mean_prob':              float(probs_arr.mean()),
            'max_prob':               float(probs_arr.max()),
            'pred_malicious_fraction': float(preds_arr.mean()),
            'final_pred':             1 if probs_arr.mean() > threshold else 0,
            'y_true_majority':        int(np.bincount(true_arr[true_arr >= 0]).argmax()) if (true_arr >= 0).any() else 0,
            'peak_ts':                records[int(probs_arr.argmax())]['timestamp'],
        }

    # ----------------------------------------------------------------
    # Pass 2: Individual attention extraction for detected malicious xApps
    # ----------------------------------------------------------------
    # Build a fast lookup: (xapp_id, timestamp) -> index in ego_dataset
    ego_index: Dict[tuple, int] = {}
    for i, g in enumerate(ego_dataset):
        ego_index[(g.global_xapp_id, g.global_timestamp)] = i

    malicious_xapps = [
        xid for xid, s in xapp_stats.items()
        if s['final_pred'] == 1
    ]
    # Sort by mean_prob descending
    malicious_xapps.sort(key=lambda x: -xapp_stats[x]['mean_prob'])

    attention_data: Dict[int, Dict] = {}
    for xid in malicious_xapps:
        peak_ts = xapp_stats[xid]['peak_ts']
        idx     = ego_index.get((xid, peak_ts))
        if idx is None:
            continue

        ego_g = ego_dataset[idx]
        data  = ego_g.to(device)
        batch_obj = _single_to_batch(data, device)

        with torch.no_grad():
            _, _, attn_ei, attn_w = model(
                batch_obj,
                return_embeds     = True,
                capture_attention = True,
            )

        attention_data[xid] = _summarise_attention(
            ego_g, attn_ei, attn_w
        )

    return {
        'all_preds':    all_preds,
        'xapp_stats':   xapp_stats,
        'attention_data': attention_data,
    }


# ==============================================================================
# HELPERS
# ==============================================================================
def _single_to_batch(data: Data, device: torch.device):
    """Wrap a single Data object so it looks like a 1-graph batch."""
    from torch_geometric.data import Batch
    return Batch.from_data_list([data]).to(device)


def _summarise_attention(
    ego_g:   Data,
    attn_ei: torch.Tensor,
    attn_w:  torch.Tensor,
) -> Dict[str, Any]:
    """
    For edges incoming to the root node, aggregate attention weight by node type.
    Node type derived from one-hot features:
        col 0 → xApp, col 1 → Cell, col 2 → UE
    """
    root     = ego_g.root_node
    x_feats  = ego_g.x.cpu().numpy()

    ei       = attn_ei.cpu().numpy()   # [2, E]
    aw       = attn_w.cpu().numpy()    # [E, heads] or [E]
    aw_flat  = aw.mean(axis=1) if aw.ndim == 2 else aw  # [E]

    # Edges whose *destination* is the root node
    dst_mask = ei[1] == root
    src_nodes = ei[0][dst_mask]
    src_attn  = aw_flat[dst_mask]

    type_attn = {'xapp': 0.0, 'cell': 0.0, 'ue': 0.0, 'unknown': 0.0}
    type_count = {'xapp': 0, 'cell': 0, 'ue': 0, 'unknown': 0}

    for node_idx, attn_val in zip(src_nodes, src_attn):
        feat = x_feats[node_idx]
        if feat[0] > 0.5:
            t = 'xapp'
        elif feat[1] > 0.5:
            t = 'cell'
        elif feat[2] > 0.5:
            t = 'ue'
        else:
            t = 'unknown'
        type_attn[t]  += attn_val
        type_count[t] += 1

    total = sum(type_attn.values()) or 1.0
    attn_pct = {k: round(v / total * 100, 1) for k, v in type_attn.items()}

    # Top-5 most attended neighbour nodes
    top_k = min(5, len(src_nodes))
    if top_k > 0:
        top_idx  = np.argsort(src_attn)[::-1][:top_k]
        top_nodes = src_nodes[top_idx].tolist()
        top_aw    = src_attn[top_idx].tolist()
    else:
        top_nodes, top_aw = [], []

    return {
        'peak_timestamp':  ego_g.global_timestamp,
        'attn_by_type':    attn_pct,
        'type_counts':     type_count,
        'top_node_indices': top_nodes,
        'top_attn_weights': top_aw,
        'x_feats':         x_feats,
        'root_node':       int(root),
        'num_nodes':       ego_g.x.size(0),
    }
