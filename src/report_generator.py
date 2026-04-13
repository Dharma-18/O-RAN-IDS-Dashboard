# ==============================================================================
# Report Generator — Attention-Derived Malicious xApp Report
# ==============================================================================
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from typing import Any, Dict


def generate_report(inference_results: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """
    Build a comprehensive report from inference results.

    Returns a dict with:
      summary       : overall statistics
      xapp_table    : list of per-xApp status rows
      malicious_list: sorted list of detected malicious xApps with detail
      confusion     : 2×2 confusion matrix
      metrics       : precision, recall, F1, accuracy
      attention_narratives: textual explanations per malicious xApp
      timeline_matrix: 2-D array [xapp × timestamp] of malicious probability
    """
    all_preds    = inference_results['all_preds']
    xapp_stats   = inference_results['xapp_stats']
    attn_data    = inference_results['attention_data']
    num_xapps    = config['topology']['num_xapps']

    # ----------------------------------------------------------------
    # Ground truth and prediction arrays
    # ----------------------------------------------------------------
    all_true, all_pred = [], []
    for rec in all_preds:
        if rec['y_true'] >= 0:
            all_true.append(rec['y_true'])
            all_pred.append(rec['pred'])

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rcl  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()

    # ----------------------------------------------------------------
    # Per-xApp table
    # ----------------------------------------------------------------
    xapp_table = []
    for xid in range(num_xapps):
        if xid not in xapp_stats:
            continue
        s = xapp_stats[xid]
        xapp_table.append({
            'xapp_id':       xid,
            'status':        '🔴 Malicious' if s['final_pred'] == 1 else '🟢 Benign',
            'confidence':    f"{s['mean_prob'] * 100:.1f}%",
            'conf_raw':      s['mean_prob'],
            'mal_fraction':  f"{s['pred_malicious_fraction'] * 100:.1f}%",
            'ground_truth':  '⚠️ Malicious' if s['y_true_majority'] == 1 else '✅ Benign',
            'correct':       s['final_pred'] == s['y_true_majority'],
            'peak_ts':       s['peak_ts'],
        })

    malicious_list = sorted(
        [r for r in xapp_table if r['status'].startswith('🔴')],
        key=lambda r: -r['conf_raw'],
    )

    # ----------------------------------------------------------------
    # Attention narratives
    # ----------------------------------------------------------------
    attention_narratives = {}
    for xid, ad in attn_data.items():
        att = ad['attn_by_type']
        top_type = max(att, key=att.get)
        tc       = ad['type_counts']
        narrative = (
            f"xApp-{xid} was flagged with {xapp_stats[xid]['mean_prob']*100:.1f}% confidence.\n"
            f"The model's attention at peak timestamp {ad['peak_timestamp']} focused primarily on "
            f"**{top_type.upper()} nodes** ({att[top_type]:.1f}% of total attention weight).\n"
        )
        if top_type == 'ue':
            n_ue = tc['ue']
            narrative += (
                f"It concentrated on {n_ue} UE node(s), consistent with targeted interference "
                f"or selective resource manipulation against specific user equipment."
            )
        elif top_type == 'cell':
            narrative += (
                f"It focused on cell-infrastructure nodes, suggesting possible "
                f"handover manipulation or load-injection attacks."
            )
        elif top_type == 'xapp':
            narrative += (
                f"Unusually high cross-xApp attention may indicate coordinated "
                f"collusion or lateral movement among compromised xApps."
            )
        attention_narratives[xid] = narrative

    # ----------------------------------------------------------------
    # Timeline matrix [xapp_id × timestamp] → mean prob per cell
    # ----------------------------------------------------------------
    ts_set  = sorted({r['timestamp'] for r in all_preds})
    ts_map  = {t: i for i, t in enumerate(ts_set)}
    xid_set = sorted(xapp_stats.keys())

    matrix = np.zeros((len(xid_set), len(ts_set)), dtype=np.float32)
    for rec in all_preds:
        xi = xid_set.index(rec['xapp_id']) if rec['xapp_id'] in xid_set else -1
        ti = ts_map.get(rec['timestamp'], -1)
        if xi >= 0 and ti >= 0:
            matrix[xi, ti] = rec['prob']

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    summary = {
        'total_xapps':          num_xapps,
        'total_ego_graphs':     len(all_preds),
        'detected_malicious':   len(malicious_list),
        'detected_benign':      num_xapps - len(malicious_list),
        'accuracy':             f"{acc * 100:.2f}%",
        'precision':            f"{prec * 100:.2f}%",
        'recall':               f"{rcl * 100:.2f}%",
        'f1_score':             f"{f1 * 100:.2f}%",
        'acc_raw':              acc,
        'prec_raw':             prec,
        'rec_raw':              rcl,
        'f1_raw':               f1,
    }

    return {
        'summary':               summary,
        'xapp_table':            xapp_table,
        'malicious_list':        malicious_list,
        'confusion':             cm,
        'attention_data':        attn_data,
        'attention_narratives':  attention_narratives,
        'timeline_matrix':       matrix,
        'timeline_xapps':        xid_set,
        'timeline_timestamps':   ts_set,
    }
