# ==============================================================================
# O-RAN xApp Intrusion Detection System — Streamlit Dashboard
# ==============================================================================
import streamlit as st
import time
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Source modules ─────────────────────────────────────────────────────────────
from src.des_generator import DatasetGenerator, DEFAULT_CONFIG
from src.graph_builder import GlobalGraphBuilder, generate_ego_graphs
from src.model import AttentionGuidedEgoGAT
from src.trainer import train as train_model
from src.inference import run_inference
from src.report_generator import generate_report
from src.visualizer import (
    create_global_graph_figure,
    create_ego_graph_figure,
    create_training_curves,
    create_timeline_heatmap,
    create_attention_breakdown,
    create_confusion_matrix_figure,
)

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="O-RAN xApp IDS — Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS — Dark Cyberpunk Theme
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #263548;
    --accent-blue: #38bdf8;
    --accent-purple: #a855f7;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-orange: #f97316;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border: #334155;
    --glow-blue: 0 0 20px rgba(56,189,248,0.15);
    --glow-purple: 0 0 20px rgba(168,85,247,0.15);
    --glow-red: 0 0 20px rgba(239,68,68,0.2);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-secondary) !important;
    border: none !important;
    background: transparent !important;
    padding: 0.8rem 1.2rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
    text-shadow: 0 0 8px rgba(56,189,248,0.4);
}
[data-testid="stTabs"] button:hover {
    color: var(--text-primary) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: var(--glow-blue) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--accent-blue) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-size: 0.75rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

/* Select boxes & inputs */
[data-testid="stSelectbox"], .stSlider, .stNumberInput {
    font-family: 'Inter', sans-serif !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7, #38bdf8) !important;
    border-radius: 8px !important;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.hero-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #a855f7, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.01em;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.25rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-red {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
}
.badge-green {
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.3);
}

/* Info cards */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    box-shadow: var(--glow-blue);
}
.info-card h4 {
    color: var(--accent-blue);
    margin: 0 0 0.4rem;
    font-weight: 700;
    font-size: 1rem;
}
.info-card p {
    color: var(--text-secondary);
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.55;
}

/* Narrative blocks */
.narrative-block {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(168,85,247,0.08));
    border-left: 4px solid var(--accent-red);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: var(--text-primary);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HERO HEADER
# ==============================================================================
st.markdown("""
<div class="hero-header">
    <h1>🛡️ O-RAN xApp Intrusion Detection System</h1>
    <p>Real-Time Threat Intelligence • Graph Neural Networks • Attention-Based Explainability</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INIT
# ==============================================================================
defaults = {
    'simulation_run':    False,
    'df':                None,
    'global_graphs':     None,
    'ego_dataset':       None,
    'model':             None,
    'history':           None,
    'inference_results': None,
    'report':            None,
    'config':            None,
    'sim_log':           [],
    'phase':             'idle',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==============================================================================
# SIDEBAR — SIMULATION CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")
    st.markdown("---")

    st.markdown("##### 📡 Network Topology")
    num_xapps = st.slider("Number of xApps", 5, 50, 15, step=5,
                          help="Total xApp instances in the O-RAN network")
    num_ues   = st.slider("Number of UEs",   50, 500, 200, step=50,
                          help="User Equipment (mobile devices)")
    num_cells = st.slider("Number of Cells",  3, 20, 5,
                          help="Cell tower base stations")

    st.markdown("##### ⏱️ Simulation Time")
    num_timesteps = st.slider("Timesteps", 100, 1000, 300, step=50,
                              help="Total simulation duration")
    window_size   = st.slider("Window Size", 10, 100, 50, step=10,
                              help="Aggregation window for deltas")

    st.markdown("##### 🧠 Model Hyperparameters")
    use_pretrained   = st.toggle("Use Pre-Trained Model (Fast)", value=True, help="Load offline-trained weights instead of training live")
    pretrain_epochs  = st.slider("Contrastive Pre-train Epochs", 1, 20, 5, disabled=use_pretrained)
    finetune_epochs  = st.slider("Supervised Fine-tune Epochs",  2, 30, 10, disabled=use_pretrained)
    learning_rate    = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, disabled=use_pretrained)
    hidden_channels  = st.selectbox("Hidden Channels", [32, 64, 128], index=0, disabled=use_pretrained)
    heads            = st.selectbox("Attention Heads",  [2, 4, 8], index=1, disabled=use_pretrained)
    batch_size       = st.selectbox("Batch Size",  [16, 32, 64, 128], index=2, disabled=use_pretrained)

    st.markdown("---")

    import time as pytime
    run_config = {
        **DEFAULT_CONFIG,
        "seed": int(pytime.time()),
        "use_pretrained": use_pretrained,
        "topology": {
            **DEFAULT_CONFIG["topology"],
            "num_xapps": num_xapps,
            "num_ues":   num_ues,
            "num_cells": num_cells,
        },
        "time": {
            "num_timesteps": num_timesteps,
            "window_size":   window_size,
        },
    }


# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================
def run_full_simulation():
    """Executes the entire pipeline, updating Streamlit state at each stage."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = run_config
    st.session_state['config'] = config
    log = st.session_state['sim_log']

    with st.status("🚀 Running Full Simulation Pipeline...", expanded=True) as status:
        # ────────────────────────────────────────────────────────────────
        # STEP 1: Data Generation
        # ────────────────────────────────────────────────────────────────
        st.write("**Step 1/6** — Generating synthetic DES dataset...")
        t0 = time.time()
        gen = DatasetGenerator(config)
        df  = gen.generate()
        dt  = time.time() - t0
        st.session_state['df'] = df
        log.append(f"✅ Generated {len(df):,} rows in {dt:.1f}s")
        st.write(f"✅ {len(df):,} rows generated ({dt:.1f}s)")

        # ────────────────────────────────────────────────────────────────
        # STEP 2: Build Global Graphs
        # ────────────────────────────────────────────────────────────────
        st.write("**Step 2/6** — Building global graphs per timestamp...")
        t0    = time.time()
        builder = GlobalGraphBuilder(config)
        builder.fit(df)
        global_graphs = builder.build_window_graphs(df)
        dt    = time.time() - t0
        st.session_state['global_graphs'] = global_graphs
        log.append(f"✅ {len(global_graphs)} global graphs ({dt:.1f}s)")
        st.write(f"✅ {len(global_graphs)} global graphs ({dt:.1f}s)")

        # ────────────────────────────────────────────────────────────────
        # STEP 3: Extract Ego Graphs
        # ────────────────────────────────────────────────────────────────
        st.write("**Step 3/6** — Extracting ego graphs for each xApp...")
        t0 = time.time()
        ego_dataset = generate_ego_graphs(global_graphs, num_xapps=config["topology"]["num_xapps"])
        dt = time.time() - t0
        st.session_state['ego_dataset'] = ego_dataset
        log.append(f"✅ {len(ego_dataset):,} ego graphs ({dt:.1f}s)")
        st.write(f"✅ {len(ego_dataset):,} ego graphs ({dt:.1f}s)")

        # ────────────────────────────────────────────────────────────────
        # STEP 4: Train / Load Model
        # ────────────────────────────────────────────────────────────────
        opt_thresh = 0.5
        if config.get("use_pretrained", True):
            st.write("**Step 4/6** — Loading pre-trained AttentionGuidedEgoGAT model...")
            import glob
            model_files = glob.glob("model_split*.pkl")
            
            if not model_files:
                st.warning("No pre-trained models found. Falling back to live training.")
                config["use_pretrained"] = False
            else:
                t0 = time.time()
                pretrained_file = next((m for m in model_files if "weighted" in m), model_files[0])
                checkpoint = torch.load(pretrained_file, map_location=device, weights_only=False)
                
                model = AttentionGuidedEgoGAT(
                    num_node_features=12,
                    hidden_channels=32, 
                    heads=4,
                    num_classes=1,
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                opt_thresh = checkpoint.get('opt_thresh', 0.5)
                
                dt = time.time() - t0
                st.session_state['model'] = model
                st.session_state['history'] = None  # Live training history unavailable
                
                log.append(f"✅ Pre-trained '{pretrained_file}' loaded ({dt:.1f}s)")
                st.write(f"✅ Loaded '{pretrained_file}' | Threshold: **{opt_thresh:.3f}** ({dt:.1f}s)")

        if not config.get("use_pretrained", True):
            st.write(f"**Step 4/6** — Training model ({pretrain_epochs}+{finetune_epochs} epochs)...")
            t0 = time.time()
            model = AttentionGuidedEgoGAT(
                num_node_features=12,
                hidden_channels=hidden_channels,
                heads=heads,
                num_classes=1,
                dropout=0.5,
                edge_dropout=0.2,
            )
            model, history = train_model(
                model           = model,
                ego_dataset     = ego_dataset,
                device          = device,
                pretrain_epochs = pretrain_epochs,
                finetune_epochs = finetune_epochs,
                lr              = learning_rate,
                batch_size      = batch_size,
            )
            dt = time.time() - t0
            st.session_state['model']   = model
            st.session_state['history'] = history
            final_acc = history['finetune_acc'][-1] if history['finetune_acc'] else 0
            log.append(f"✅ Model trained — acc={final_acc:.2%} ({dt:.1f}s)")
            st.write(f"✅ Training complete — accuracy: {final_acc:.2%} ({dt:.1f}s)")

        # ────────────────────────────────────────────────────────────────
        # STEP 5: Run Inference
        # ────────────────────────────────────────────────────────────────
        st.write("**Step 5/6** — Running inference + extracting attention weights...")
        t0 = time.time()
        results = run_inference(model, ego_dataset, device, batch_size=batch_size, threshold=opt_thresh)
        dt = time.time() - t0
        st.session_state['inference_results'] = results
        n_mal = sum(1 for s in results['xapp_stats'].values() if s['final_pred'] == 1)
        log.append(f"✅ Inference complete — {n_mal} malicious detected ({dt:.1f}s)")
        st.write(f"✅ {n_mal} malicious xApps detected ({dt:.1f}s)")

        # ────────────────────────────────────────────────────────────────
        # STEP 6: Generate Report
        # ────────────────────────────────────────────────────────────────
        st.write("**Step 6/6** — Generating threat intelligence report...")
        t0     = time.time()
        report = generate_report(results, config)
        dt     = time.time() - t0
        st.session_state['report'] = report
        log.append(f"✅ Report generated ({dt:.1f}s)")
        st.write(f"✅ Report generated ({dt:.1f}s)")

        st.session_state['simulation_run'] = True
        status.update(label="✅ Simulation Complete!", state="complete", expanded=False)


# ==============================================================================
# TABS
# ==============================================================================
tab_sim, tab_global, tab_ego, tab_train, tab_report = st.tabs([
    "🚀 Simulation",
    "🌐 Global Graph",
    "🕸️ Ego Graph",
    "🧠 Training",
    "📊 Threat Report",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: SIMULATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_sim:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🔬 Simulation Pipeline</h4>
            <p>
                Generates a synthetic O-RAN network dataset, builds heterogeneous global graphs
                per timestep, extracts ego subgraphs around each xApp, trains a GATv2-based GNN
                with contrastive pre-training, runs inference, and produces a threat report
                derived from the model's attention weights.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        bcol1, bcol2 = st.columns([1, 3])
        with bcol1:
            run_btn = st.button("▶ Start Simulation", use_container_width=True,
                                type="primary")
        with bcol2:
            if st.session_state['simulation_run']:
                st.success("✅ Simulation completed — explore the tabs above!")

        if run_btn:
            run_full_simulation()
            st.rerun()

    with col2:
        st.markdown("##### 📋 Current Configuration")
        cfg_df = pd.DataFrame({
            "Parameter": ["xApps", "UEs", "Cells", "Timesteps", "Window",
                          "Pre-train Epochs", "Fine-tune Epochs", "LR",
                          "Hidden Ch.", "Heads", "Batch Size", "Device"],
            "Value": [
                str(num_xapps), str(num_ues), str(num_cells), str(num_timesteps), str(window_size),
                str(pretrain_epochs), str(finetune_epochs), str(learning_rate),
                str(hidden_channels), str(heads), str(batch_size),
                "CUDA" if torch.cuda.is_available() else "CPU"
            ]
        })
        st.dataframe(cfg_df, use_container_width=True, hide_index=True, height=440)

    # Simulation Log
    if st.session_state['sim_log']:
        with st.expander("📜 Simulation Log", expanded=False):
            for entry in st.session_state['sim_log']:
                st.text(entry)

    # Dataset preview
    if st.session_state['df'] is not None:
        with st.expander("📊 Dataset Preview (first 200 rows)", expanded=False):
            st.dataframe(st.session_state['df'].head(200), use_container_width=True, height=300)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: GLOBAL GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_global:
    if st.session_state['global_graphs'] is None:
        st.info("👈 Run the simulation first to view global graphs.")
    else:
        graphs = st.session_state['global_graphs']
        config = st.session_state['config']

        col1, col2 = st.columns([3, 1])
        with col2:
            ts_options = [g.timestamp for g in graphs]
            selected_ts = st.selectbox(
                "Select Timestamp",
                ts_options,
                index=min(10, len(ts_options) - 1),
                key="global_ts",
            )
            idx     = ts_options.index(selected_ts)
            g_data  = graphs[idx]

            n_mal = int((g_data.y[:config["topology"]["num_xapps"]] == 1).sum())
            n_ben = config["topology"]["num_xapps"] - n_mal
            st.metric("Total Nodes", f"{g_data.x.size(0):,}")
            st.metric("Total Edges", f"{g_data.edge_index.size(1):,}")
            st.metric("Malicious xApps", n_mal)
            st.metric("Benign xApps",    n_ben)

        with col1:
            fig = create_global_graph_figure(
                g_data,
                config["topology"]["num_xapps"],
                config["topology"]["num_cells"],
            )
            st.plotly_chart(fig, use_container_width=True, key="global_fig")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: EGO GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ego:
    if st.session_state['ego_dataset'] is None:
        st.info("👈 Run the simulation first to view ego graphs.")
    else:
        ego_dataset = st.session_state['ego_dataset']
        config      = st.session_state['config']
        n_xapps     = config["topology"]["num_xapps"]

        ts_list = sorted(list(set([int(g.global_timestamp) for g in ego_dataset])))
        
        st.markdown("### 🕸️ Ego-Graph Viewport")
        view_mode = st.radio("Display Mode:", ["Single xApp Focus", f"View All {n_xapps} xApps (Grid)"], horizontal=True)
        sel_ego_ts = st.select_slider("⏱️ Timeline (Sweep Timestamp)", options=ts_list)
        
        st.markdown("---")

        if "Single" in view_mode:
            col1, col2 = st.columns([3, 1])

            with col2:
                sel_xapp = st.selectbox("Select xApp", list(range(n_xapps)), key="ego_xapp")

                # Find ego graph for this xApp at this timestamp
                ego_g_cands = [g for g in ego_dataset if g.global_xapp_id == sel_xapp and g.global_timestamp == sel_ego_ts]
                
                if ego_g_cands:
                    ego_g = ego_g_cands[0]
                    label_str = "🔴 MALICIOUS" if ego_g.y.item() == 1 else "🟢 BENIGN"
                    st.markdown(f"**Ground Truth:** {label_str}")
                    st.metric("Nodes in Subgraph", ego_g.x.size(0))
                    st.metric("Edges in Subgraph", ego_g.edge_index.size(1))
                    
                    if st.session_state['inference_results']:
                        stats = st.session_state['inference_results']['xapp_stats']
                        if sel_xapp in stats:
                            s = stats[sel_xapp]
                            pred_str = "🔴 MALICIOUS" if s['final_pred'] == 1 else "🟢 BENIGN"
                            st.markdown(f"**Model Prediction:** {pred_str}")
                            st.metric("Confidence", f"{s['mean_prob']*100:.1f}%")
                else:
                    st.warning("No ego graph available at this timestep.")
                    ego_g = None

            with col1:
                if ego_g:
                    fig = create_ego_graph_figure(ego_g)
                    st.plotly_chart(fig, use_container_width=True, key="ego_fig")

        else:
            # Grid View for ALL xApps
            st.markdown(f"##### Grid View: Time = {sel_ego_ts}")
            cols = st.columns(3)
            grid_egos = [g for g in ego_dataset if g.global_timestamp == sel_ego_ts]
            
            for i, ego_g in enumerate(grid_egos):
                xapp_id = int(ego_g.global_xapp_id)
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"**xApp {xapp_id}**")
                        fig = create_ego_graph_figure(ego_g)
                        # Make plots slightly smaller for grid
                        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True, key=f"grid_fig_{xapp_id}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_train:
    if st.session_state['history'] is None:
        st.info("👈 Run the simulation first to view training metrics.")
    else:
        history = st.session_state['history']

        # KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            init_loss = history['pretrain_loss'][0] if history['pretrain_loss'] else 0
            st.metric("Initial Contrastive Loss", f"{init_loss:.3f}")
        with m2:
            final_loss = history['pretrain_loss'][-1] if history['pretrain_loss'] else 0
            st.metric("Final Contrastive Loss", f"{final_loss:.3f}")
        with m3:
            final_bce = history['finetune_loss'][-1] if history['finetune_loss'] else 0
            st.metric("Final BCE Loss", f"{final_bce:.3f}")
        with m4:
            final_acc = history['finetune_acc'][-1] if history['finetune_acc'] else 0
            st.metric("Final Accuracy", f"{final_acc:.2%}")

        st.markdown("")
        fig = create_training_curves(history)
        st.plotly_chart(fig, use_container_width=True, key="train_fig")

        # Epoch table
        with st.expander("📋 Epoch-by-Epoch Details"):
            rows = []
            for i, l in enumerate(history.get('pretrain_loss', [])):
                rows.append({"Phase": "Pre-train", "Epoch": i+1, "Loss": f"{l:.4f}", "Accuracy": "—"})
            n_pre = len(history.get('pretrain_loss', []))
            for i, (l, a) in enumerate(zip(
                    history.get('finetune_loss', []),
                    history.get('finetune_acc', []))):
                rows.append({"Phase": "Fine-tune", "Epoch": n_pre+i+1, "Loss": f"{l:.4f}", "Accuracy": f"{a:.2%}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: THREAT REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_report:
    if st.session_state['report'] is None:
        st.info("👈 Run the simulation first to generate the threat report.")
    else:
        report = st.session_state['report']
        summary = report['summary']

        # ── KPI Row ──
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total xApps", summary['total_xapps'])
        with k2:
            st.metric("Detected Malicious", summary['detected_malicious'])
        with k3:
            st.metric("Detection Accuracy", summary['accuracy'])
        with k4:
            st.metric("Precision", summary['precision'])
        with k5:
            st.metric("F1 Score", summary['f1_score'])

        st.markdown("")

        # ── Main layout: table + confusion ──
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### 📋 xApp Classification Results")
            table_df = pd.DataFrame(report['xapp_table'])
            display_df = table_df[['xapp_id', 'status', 'confidence', 'mal_fraction', 'ground_truth', 'correct']].copy()
            display_df.columns = ['xApp ID', 'Prediction', 'Confidence', 'Malicious %', 'Ground Truth', 'Correct']
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=min(400, 40 * len(display_df)),
            )

        with col_right:
            st.markdown("#### 🧮 Confusion Matrix")
            if report['confusion'] and len(report['confusion']) == 2:
                fig_cm = create_confusion_matrix_figure(report['confusion'])
                st.plotly_chart(fig_cm, use_container_width=True, key="cm_fig")
            else:
                st.warning("Confusion matrix not available.")

        st.markdown("---")

        # ── Timeline Heatmap ──
        st.markdown("#### 🔥 Threat Timeline Heatmap")
        st.caption("Each cell shows the model's malicious probability for a specific xApp at a given timestamp.")
        fig_hm = create_timeline_heatmap(
            report['timeline_matrix'],
            report['timeline_xapps'],
            report['timeline_timestamps'],
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="hm_fig")

        st.markdown("---")

        # ── Attention Analysis ──
        st.markdown("#### 🔬 Attention-Weight Analysis (Explainability)")

        if report['attention_data']:
            fig_ab = create_attention_breakdown(report['attention_data'])
            st.plotly_chart(fig_ab, use_container_width=True, key="ab_fig")

            st.markdown("")
            st.markdown("##### 🗣️ Attention Narratives — Why Each xApp Was Flagged")

            for xid, narrative in sorted(report['attention_narratives'].items()):
                st.markdown(f"""<div class="narrative-block">
                    <strong>⚠️ xApp-{xid}</strong><br>{narrative}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No malicious xApps were detected — no attention data to display for flagged entities.")

        st.markdown("---")

        # ── Malicious xApps Summary ──
        if report['malicious_list']:
            st.markdown("#### 🚨 Detected Malicious xApps — Summary")
            for entry in report['malicious_list']:
                xid = entry['xapp_id']
                badge_cls = "badge-red"
                st.markdown(f"""
                <div class="info-card" style="border-left: 4px solid #ef4444;">
                    <h4>xApp-{xid} <span class="status-badge {badge_cls}">THREAT DETECTED</span></h4>
                    <p>
                        <strong>Confidence:</strong> {entry['confidence']} &nbsp;|&nbsp;
                        <strong>Malicious Fraction:</strong> {entry['mal_fraction']} &nbsp;|&nbsp;
                        <strong>Peak Timestamp:</strong> {entry['peak_ts']} &nbsp;|&nbsp;
                        <strong>Ground Truth:</strong> {entry['ground_truth']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No malicious xApps detected in this simulation run.")
