<div align="center">
  <h1>🛡️ O-RAN IDS Dashboard</h1>
  <p><b>Real-Time Intrusion Detection System for O-RAN using Graph Neural Networks</b></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-Geometric-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

<br>

## 📌 Overview
The **O-RAN Intrusion Detection System (IDS) Dashboard** is a sophisticated full-stack cybersecurity framework designed to detect, analyze, and explain anomalous adversarial actions (such as QoS manipulation, traffic shaping drops, and unauthorized handovers) within Open RAN architecture. 

It accomplishes this by harnessing **Discrete Event Simulation (DES)** telemetry and passing topological relationships into an **Attention-Guided Graph Neural Network (GNN)**, powered by PyTorch Geometric.

## ✨ Key Features
- **Vectorized O-RAN Simulation**: Generates 100k+ telemetry matrices dynamically mapping complex 5G network relationships in milliseconds using Numpy/Pandas matrices.
- **Explainable AI (XAI)**: Native Graph Attention architecture (`GATv2Conv`) seamlessly intercepts and maps relational node weights—offering deep-text narratives explaining *why* an xApp was compromised.
- **Offline ML Pipeline**: Employs completely deterministic `.pkl` structural binary state serialization using Contrastive Pre-Training + Task-Specific Supervised Fine-Tuning.
- **Cybersecurity Streamlit Interface**: Offers a premium, cyberpunk-aesthetic analytical dashboard highlighting real-time Confusion Matrices, Ego-Graph Timelines, and Grid Network views securely via `Plotly`.

---

## 📂 Project Structure
```text
├── src/
│   ├── des_generator.py      # Vectorized Open-RAN Data Synthesizer
│   ├── graph_builder.py      # Global & Ego-Graph PyG Constructors
│   ├── inference.py          # Dual-Pass Inference & Attention Extractor
│   ├── model.py              # EgoGAT & AttentionGuided Architecture 
│   ├── report_generator.py   # XAI Explainability Narrative Compiler 
│   ├── trainer.py            # Offline GNN Evaluation Loop 
│   └── visualizer.py         # Plotly Heatmap & Matrix Grid Logic
├── app.py                    # Main Streamlit Analytical Dashboard
├── generate_models.py        # Standalone ML Engine Pipeline
└── README.md                 # Project Documentation
```

## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Dharma-18/CIP_WEB.git
cd CIP_WEB
```

**2. Install runtime dependencies:**
*(Ensure you have a modern version of PyTorch installed for your specific CUDA/CPU layout prior to running)*
```bash
pip install torch torch-geometric pandas numpy scikit-learn streamlit plotly
```

## 🚀 Usage

### 1. Model Compilation (Optional/Offline Setup)
Our pre-trained `.pkl` checkpoints dynamically load instantly. To manually rewrite, expand, or adjust mathematical thresholds across a new generated dataset:
```bash
python generate_models.py
```
*This will construct `model_split60_std.pkl` and `model_split60_weighted.pkl` in the root environment.*

### 2. Launch Cyber Intelligence Dashboard
To spin up the web application and visualize network payloads securely via the browser:
```bash
streamlit run app.py
```

---

## 🔮 Expected Metrics
On highly synthetic DES traces aggregating `15` xApps testing millions of localized edge-networks, our contrastive Attention-Layer secures:
- **Detection Accuracy**: ~94%+
- **F1 Validation**: ~87%+
- **Precision Thresholding**: ~96%+
