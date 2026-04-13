# ==============================================================================
# Visualizer — Plotly-Based Graph & Training Visualizations
# ==============================================================================
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Dict, List, Any, Optional


# ── Colour palette ─────────────────────────────────────────────────────────────
_C = {
    'benign':    '#f59e0b',   # amber
    'malicious': '#ef4444',   # red
    'cell':      '#22c55e',   # green
    'ue':        '#38bdf8',   # sky-blue
    'ego':       '#a855f7',   # purple (root node)
    'bg':        '#0f172a',
    'grid':      '#1e293b',
    'text':      '#e2e8f0',
}


# ==============================================================================
# 1. GLOBAL GRAPH (Plotly)
# ==============================================================================
def create_global_graph_figure(
    data:         Data,
    num_xapps:    int,
    num_cells:    int,
    max_ue_shown: int = 80,
) -> go.Figure:
    """
    Render a global graph snapshot as an interactive Plotly figure.
    Large UE sets are sampled for performance.
    """
    G        = to_networkx(data, to_undirected=True)
    x_feats  = data.x.numpy()
    y_labels = data.y.numpy()

    # ---- Node layout ----
    pos = nx.spring_layout(G, seed=42, k=0.3, iterations=60)

    # ---- Classify nodes ----
    xapp_benign, xapp_mal, cells, ues_shown = [], [], [], []
    total_ues = 0

    for i in G.nodes():
        f = x_feats[i]
        if f[0] > 0.5:  # xApp
            if y_labels[i] == 1:
                xapp_mal.append(i)
            else:
                xapp_benign.append(i)
        elif f[1] > 0.5:  # Cell
            cells.append(i)
        elif f[2] > 0.5:  # UE
            total_ues += 1
            ues_shown.append(i)

    # Sample UEs if too many
    if len(ues_shown) > max_ue_shown:
        rng       = np.random.default_rng(0)
        ues_shown = rng.choice(ues_shown, max_ue_shown, replace=False).tolist()

    # ---- Edge trace ----
    edge_x, edge_y = [], []
    for u, v in G.edges():
        xu, yu = pos[u]; xv, yv = pos[v]
        edge_x += [xu, xv, None]; edge_y += [yu, yv, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#334155'),
        hoverinfo='none',
        showlegend=False,
    )

    def node_trace(node_ids, color, name, size=14, symbol='circle'):
        xs = [pos[n][0] for n in node_ids]
        ys = [pos[n][1] for n in node_ids]
        return go.Scatter(
            x=xs, y=ys,
            mode='markers+text' if size >= 14 else 'markers',
            marker=dict(size=size, color=color, line=dict(width=1, color='#0f172a')),
            text=[f"xApp-{n}" if n < num_xapps else (f"Cell-{n - num_xapps}" if n < num_xapps + num_cells else "") for n in node_ids],
            textposition='top center',
            textfont=dict(size=8, color=_C['text']),
            name=name,
            hovertext=[_node_hover(n, x_feats, y_labels, num_xapps, num_cells) for n in node_ids],
            hoverinfo='text',
            marker_symbol=symbol,
        )

    fig = go.Figure(data=[
        edge_trace,
        node_trace(ues_shown,   _C['ue'],       f'UE (showing {len(ues_shown)}/{total_ues})', size=6),
        node_trace(cells,       _C['cell'],      'Cell Tower', size=20, symbol='diamond'),
        node_trace(xapp_benign, _C['benign'],    'Benign xApp', size=16),
        node_trace(xapp_mal,    _C['malicious'], 'Malicious xApp', size=18, symbol='x'),
    ])

    fig.update_layout(
        paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
        font=dict(color=_C['text']),
        showlegend=True,
        legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        hovermode='closest',
    )
    return fig


def _node_hover(n, x_feats, y_labels, num_xapps, num_cells):
    f = x_feats[n]
    if f[0] > 0.5:
        label = 'Malicious' if y_labels[n] == 1 else 'Benign'
        return f"<b>xApp-{n}</b><br>Status: {label}<br>CPU: {f[3]:.2f} | Mem: {f[4]:.2f}"
    elif f[1] > 0.5:
        cell_id = n - num_xapps
        return f"<b>Cell-{cell_id}</b><br>Load: {f[7]:.2f} | UE Count: {f[8]:.2f}"
    elif f[2] > 0.5:
        ue_id = n - num_xapps - num_cells
        return f"<b>UE-{ue_id}</b><br>RSRP: {f[9]:.2f}<br>SINR: {f[10]:.2f} | TP: {f[11]:.2f}"
    return f"Node-{n}"


# ==============================================================================
# 2. EGO GRAPH (Plotly)
# ==============================================================================
def create_ego_graph_figure(ego_data: Data) -> go.Figure:
    G       = to_networkx(ego_data, to_undirected=True)
    x_feats = ego_data.x.numpy()
    root    = int(ego_data.root_node)

    pos = nx.spring_layout(G, seed=42, k=0.5)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        xu, yu = pos[u]; xv, yv = pos[v]
        edge_x += [xu, xv, None]; edge_y += [yu, yv, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.2, color='#334155'),
        hoverinfo='none', showlegend=False,
    )

    def ego_node_trace(ids, color, name, size, symbol='circle'):
        return go.Scatter(
            x=[pos[i][0] for i in ids], y=[pos[i][1] for i in ids],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
            text=[_ego_label(i, x_feats, root) for i in ids],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='monospace'),
            name=name,
            hovertext=[_ego_hover(i, x_feats, root) for i in ids],
            hoverinfo='text',
            marker_symbol=symbol,
        )

    # Categorise
    root_n, xapp_n, cell_n, ue_n = [], [], [], []
    for i in G.nodes():
        if i == root:                 root_n.append(i)
        elif x_feats[i, 0] > 0.5:    xapp_n.append(i)
        elif x_feats[i, 1] > 0.5:    cell_n.append(i)
        elif x_feats[i, 2] > 0.5:    ue_n.append(i)

    traces = [edge_trace]
    if ue_n:    traces.append(ego_node_trace(ue_n,    _C['ue'],       'UE',       22))
    if cell_n:  traces.append(ego_node_trace(cell_n,  _C['cell'],     'Cell',     30, 'diamond'))
    if xapp_n:  traces.append(ego_node_trace(xapp_n,  _C['benign'],   'xApp',     26))
    if root_n:  traces.append(ego_node_trace(root_n,  _C['ego'],      'EGO xApp', 38, 'star'))

    status = "MALICIOUS" if ego_data.y.item() == 1 else "BENIGN"
    col    = _C['malicious'] if status == "MALICIOUS" else _C['cell']

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
        font=dict(color=_C['text']),
        showlegend=True,
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(
            text=f"Ego Graph — xApp-{ego_data.global_xapp_id}  |  <span style='color:{col}'>{status}</span>",
            font=dict(size=14, color=_C['text']),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=480,
    )
    return fig


def _ego_label(i, x_feats, root):
    if i == root:          return "EGO"
    elif x_feats[i,0]>0.5: return "xApp"
    elif x_feats[i,1]>0.5: return "Cell"
    elif x_feats[i,2]>0.5: return "UE"
    return "?"


def _ego_hover(i, x_feats, root):
    prefix = "★ ROOT — " if i == root else ""
    f = x_feats[i]
    if f[0] > 0.5:  return f"{prefix}<b>xApp node {i}</b><br>CPU:{f[3]:.2f} Mem:{f[4]:.2f}"
    elif f[1]>0.5:  return f"{prefix}<b>Cell node {i}</b><br>Load:{f[7]:.2f} UEcnt:{f[8]:.2f}"
    elif f[2]>0.5:  return f"{prefix}<b>UE node {i}</b><br>RSRP:{f[9]:.2f} SINR:{f[10]:.2f} TP:{f[11]:.2f}"
    return f"Node-{i}"


# ==============================================================================
# 3. TRAINING CURVES
# ==============================================================================
def create_training_curves(history: Dict[str, List]) -> go.Figure:
    fig = go.Figure()

    if history.get('pretrain_loss'):
        epochs = list(range(1, len(history['pretrain_loss']) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['pretrain_loss'],
            name='Contrastive Loss (Pre-train)',
            line=dict(color='#38bdf8', width=2, dash='dot'),
        ))

    if history.get('finetune_loss'):
        n_pre  = len(history.get('pretrain_loss', []))
        epochs = list(range(n_pre + 1, n_pre + len(history['finetune_loss']) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['finetune_loss'],
            name='BCE Loss (Fine-tune)',
            line=dict(color='#f97316', width=2),
        ))

    if history.get('finetune_acc'):
        n_pre  = len(history.get('pretrain_loss', []))
        epochs = list(range(n_pre + 1, n_pre + len(history['finetune_acc']) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['finetune_acc'],
            name='Accuracy',
            line=dict(color='#22c55e', width=2),
            yaxis='y2',
        ))

    fig.update_layout(
        paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
        font=dict(color=_C['text']),
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        xaxis=dict(title='Epoch', gridcolor=_C['grid']),
        yaxis=dict(title='Loss', gridcolor=_C['grid'], range=[0, None]),
        yaxis2=dict(
            title='Accuracy', overlaying='y', side='right',
            range=[0, 1], gridcolor='rgba(0,0,0,0)',
            tickformat='.0%',
        ),
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified',
    )
    return fig


# ==============================================================================
# 4. TIMELINE HEATMAP
# ==============================================================================
def create_timeline_heatmap(
    matrix:     np.ndarray,
    xapp_ids:   List[int],
    timestamps: List[int],
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z         = matrix,
        x         = timestamps,
        y         = [f"xApp-{x}" for x in xapp_ids],
        colorscale= [[0, '#0f172a'], [0.4, '#1e3a5f'], [0.7, '#f97316'], [1.0, '#ef4444']],
        zmin=0, zmax=1,
        colorbar  = dict(
            title=dict(
                text='Malicious<br>Probability',
                font=dict(color=_C['text'])
            ),
            tickfont=dict(color=_C['text']),
        ),
        hovertemplate='Timestamp: %{x}<br>xApp: %{y}<br>Probability: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
        font=dict(color=_C['text']),
        xaxis=dict(title='Timestamp', gridcolor=_C['grid']),
        yaxis=dict(title='xApp ID',   gridcolor=_C['grid'], autorange='reversed'),
        height=max(300, 20 * len(xapp_ids)),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ==============================================================================
# 5. ATTENTION BREAKDOWN
# ==============================================================================
def create_attention_breakdown(attention_data: Dict[int, Dict]) -> go.Figure:
    if not attention_data:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
                          font=dict(color=_C['text']), height=280)
        return fig

    xapp_ids  = sorted(attention_data.keys())
    ue_vals   = [attention_data[x]['attn_by_type']['ue']      for x in xapp_ids]
    cell_vals = [attention_data[x]['attn_by_type']['cell']    for x in xapp_ids]
    xapp_vals = [attention_data[x]['attn_by_type']['xapp']    for x in xapp_ids]
    labels    = [f"xApp-{x}" for x in xapp_ids]

    fig = go.Figure(data=[
        go.Bar(name='UE Attention (%)',   x=labels, y=ue_vals,   marker_color=_C['ue']),
        go.Bar(name='Cell Attention (%)', x=labels, y=cell_vals, marker_color=_C['cell']),
        go.Bar(name='xApp Attention (%)', x=labels, y=xapp_vals, marker_color=_C['benign']),
    ])
    fig.update_layout(
        barmode='stack',
        paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
        font=dict(color=_C['text']),
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        xaxis=dict(title='Malicious xApp', gridcolor=_C['grid']),
        yaxis=dict(title='Attention (%)',  gridcolor=_C['grid']),
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x',
    )
    return fig


# ==============================================================================
# 6. CONFUSION MATRIX
# ==============================================================================
def create_confusion_matrix_figure(cm: List[List[int]]) -> go.Figure:
    labels      = ['Benign', 'Malicious']
    annotations = []
    z           = np.array(cm)
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=str(z[i, j]),
                showarrow=False,
                font=dict(color='white', size=20, family='JetBrains Mono'),
            ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels, y=labels,
        colorscale=[[0, '#0f172a'], [1, '#a855f7']],
        showscale=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=_C['bg'], plot_bgcolor=_C['bg'],
        font=dict(color=_C['text']),
        annotations=annotations,
        xaxis=dict(title='Predicted', side='bottom'),
        yaxis=dict(title='Actual', autorange='reversed'),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig
