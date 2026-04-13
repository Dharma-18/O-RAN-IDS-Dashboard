# ==============================================================================
# O-RAN IDS Dataset Generator (Optimized Vectorized Version) — Module Edition
# ==============================================================================
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict

# ==============================================================================
# LOGGING
# ==============================================================================
def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger("IDS_Gen")

# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================
DEFAULT_CONFIG = {
    "seed": 42,
    "topology": {
        "num_xapps": 15,
        "num_ues": 200,
        "num_cells": 5,
        "malicious_timeline": [
            (0.0, 0.2, 0.10),
            (0.2, 0.5, 0.30),
            (0.5, 0.8, 0.50),
            (0.8, 1.0, 0.20)
        ]
    },
    "time": {
        "num_timesteps": 300,
        "window_size": 50
    },
    "physics": {
        "rsrp_min": -115.0, "rsrp_max": -65.0,
        "sinr_min": -5.0,   "sinr_max": 30.0,
        "throughput_min": 10.0, "throughput_max": 300.0,
        "qos_score_min": 0.0,   "qos_score_max": 1.0,
        "cell_load_min": 0.1,   "cell_load_max": 0.99,
        "ue_count_per_cell_min": 5, "ue_count_per_cell_max": 50
    },
    "actions": {
        "types": [
            "power_adjust", "scheduling_change", "handover",
            "prb_allocation_adjust", "mcs_override", "beamforming_adjust",
            "traffic_shaping", "rmr_route_manipulation"
        ],
        "power_adjust_range": [-6.0, 6.0],
        "scheduling_priority_range": [1, 10],
        "mcs_override_range": [0, 28],
        "prb_allocation_range": [1, 100],
        "beamforming_index_range": [0, 63],
        "traffic_shaping_mbps_range": [10, 1000],
        "rmr_route_manipulation_values": [0, 1]
    },
    "malicious": {
        "target_ue_count": 5,
        "degradation_sinr": -8.0,
        "degradation_throughput_factor": 0.55,
        "noise_std": 1.0,
        "benign_occasional_dip_prob": 0.02
    },
    "output": {
        "data_dir": "./data"
    }
}

# ==============================================================================
# HELPERS
# ==============================================================================
_LOAD_MODES = ["low", "medium", "high", "burst"]

def _get_load_mode_vectorized(timestamps: np.ndarray, period: int = 1500) -> np.ndarray:
    indices = (timestamps // period) % len(_LOAD_MODES)
    return np.array(_LOAD_MODES)[indices]


# ==============================================================================
# PHYSICS & ACTION GENERATOR
# ==============================================================================
class PhysicsGenerator:
    def __init__(self, config: Dict[str, Any], is_malicious_matrix: np.ndarray, mal_targets: Dict[int, list]):
        self.config = config
        self.seed   = config["seed"]
        self.rng    = np.random.default_rng(self.seed)

        topo = config["topology"]
        self.num_xapps = topo["num_xapps"]
        self.num_ues   = topo["num_ues"]
        self.num_cells = topo["num_cells"]

        t_cfg = config["time"]
        self.num_timesteps = t_cfg["num_timesteps"]
        self.window_size   = t_cfg["window_size"]

        self.physics      = config["physics"]
        self.action_types = config["actions"]["types"]
        self.power_range  = config["actions"]["power_adjust_range"]
        self.sched_range  = config["actions"]["scheduling_priority_range"]
        self.mcs_range    = config["actions"]["mcs_override_range"]
        self.prb_range    = config["actions"]["prb_allocation_range"]
        self.beam_range   = config["actions"]["beamforming_index_range"]
        self.ts_range     = config["actions"]["traffic_shaping_mbps_range"]
        self.rmr_values   = config["actions"]["rmr_route_manipulation_values"]
        self.mal_config   = config["malicious"]

        self.ue_cell_map = self.rng.integers(0, self.num_cells, size=self.num_ues)
        self.is_malicious_matrix = is_malicious_matrix
        self.mal_targets = mal_targets

    def generate(self) -> pd.DataFrame:
        logger.info("Generating Physics & Action Context (Vectorized)...")
        ue_ctx  = self._generate_ue_context()
        ue_ctx  = self._apply_degradation(ue_ctx)
        actions = self._generate_actions()
        actions = self._inject_malicious_actions(actions)
        df      = self._assemble(actions, ue_ctx)
        return df

    def _smooth_series_vectorized(self, vmin, vmax, num_series, length, smoothness=0.95):
        mid  = (vmin + vmax) / 2.0
        span = (vmax - vmin) / 2.0
        noise = self.rng.normal(0, span * (1 - smoothness), size=(length, num_series))
        x = mid + noise / (1 - smoothness)
        df = pd.DataFrame(x)
        df.iloc[0, :] = self.rng.uniform(vmin, vmax, size=num_series)
        smoothed = df.ewm(alpha=1-smoothness, adjust=False).mean().values.T
        return np.clip(smoothed, vmin, vmax).flatten()

    def _generate_ue_context(self) -> pd.DataFrame:
        T = self.num_timesteps
        U = self.num_ues
        p = self.physics

        ue_ids     = np.arange(U).repeat(T)
        timestamps = np.tile(np.arange(T), U)
        cell_ids   = self.ue_cell_map[ue_ids]

        rsrp_arr = self._smooth_series_vectorized(p["rsrp_min"], p["rsrp_max"], U, T)
        sinr_arr = self._smooth_series_vectorized(p["sinr_min"], p["sinr_max"], U, T)
        tp_arr   = self._smooth_series_vectorized(p["throughput_min"], p["throughput_max"], U, T)
        qos_arr  = self._smooth_series_vectorized(p["qos_score_min"], p["qos_score_max"], U, T)

        ue_counts  = self.rng.integers(p["ue_count_per_cell_min"], p["ue_count_per_cell_max"] + 1, size=U * T)
        cell_loads = np.clip(
            0.3 + 0.03 * ue_counts + self.rng.normal(0, 0.05, size=U * T),
            p["cell_load_min"], p["cell_load_max"]
        )

        return pd.DataFrame({
            "timestamp": timestamps, "ue_id": ue_ids, "cell_id": cell_ids,
            "rsrp": np.round(rsrp_arr, 2), "sinr": np.round(sinr_arr, 2),
            "cell_load": np.round(cell_loads, 4), "ue_count": ue_counts,
            "throughput": np.round(tp_arr, 2), "qos_score": np.round(qos_arr, 4),
        })

    def _generate_actions(self) -> pd.DataFrame:
        T = self.num_timesteps
        X = self.num_xapps

        num_actions   = self.rng.integers(1, 4, size=T * X)
        total_actions = num_actions.sum()

        t_grid, x_grid = np.meshgrid(np.arange(T), np.arange(X), indexing='ij')
        timestamps = np.repeat(t_grid.flatten(), num_actions)
        xapp_ids   = np.repeat(x_grid.flatten(), num_actions)

        atypes       = self.rng.choice(self.action_types, size=total_actions)
        target_ues   = self.rng.integers(0, self.num_ues, size=total_actions)
        target_cells = self.ue_cell_map[target_ues]

        param_values = np.zeros(total_actions)
        pa_mask   = atypes == "power_adjust"
        sc_mask   = atypes == "scheduling_change"
        ho_mask   = atypes == "handover"
        mcs_mask  = atypes == "mcs_override"
        prb_mask  = atypes == "prb_allocation_adjust"
        beam_mask = atypes == "beamforming_adjust"
        ts_mask   = atypes == "traffic_shaping"
        rmr_mask  = atypes == "rmr_route_manipulation"

        param_values[pa_mask]   = self.rng.uniform(*self.power_range, size=pa_mask.sum())
        param_values[sc_mask]   = self.rng.integers(self.sched_range[0], self.sched_range[1] + 1, size=sc_mask.sum())
        param_values[ho_mask]   = self.rng.integers(0, self.num_cells, size=ho_mask.sum())
        param_values[mcs_mask]  = self.rng.integers(self.mcs_range[0], self.mcs_range[1] + 1, size=mcs_mask.sum())
        param_values[prb_mask]  = self.rng.integers(self.prb_range[0], self.prb_range[1] + 1, size=prb_mask.sum())
        param_values[beam_mask] = self.rng.integers(self.beam_range[0], self.beam_range[1] + 1, size=beam_mask.sum())
        param_values[ts_mask]   = self.rng.integers(self.ts_range[0], self.ts_range[1] + 1, size=ts_mask.sum())
        param_values[rmr_mask]  = self.rng.choice(self.rmr_values, size=rmr_mask.sum())

        return pd.DataFrame({
            "timestamp": timestamps, "xapp_id": xapp_ids, "action_type": atypes,
            "target_ue": target_ues, "target_cell": target_cells,
            "param_value": np.round(param_values, 4),
        })

    def _inject_malicious_actions(self, actions: pd.DataFrame) -> pd.DataFrame:
        actions  = actions.copy()
        mal_mask = self.is_malicious_matrix[actions["xapp_id"], actions["timestamp"]]
        mal_idx  = actions[mal_mask].index

        if len(mal_idx) > 0:
            redir_idx  = self.rng.choice(mal_idx, size=int(len(mal_idx) * 0.70), replace=False)
            target_map = np.array([self.rng.choice(self.mal_targets[x]) for x in actions.loc[redir_idx, "xapp_id"]])
            actions.loc[redir_idx, "target_ue"]   = target_map
            actions.loc[redir_idx, "target_cell"] = self.ue_cell_map[target_map]

        return actions

    def _apply_degradation(self, ue_ctx: pd.DataFrame) -> pd.DataFrame:
        mc = self.mal_config
        p  = self.physics
        T  = self.num_timesteps
        U  = self.num_ues

        ue_under_attack = np.zeros((U, T), dtype=bool)
        for xid in range(self.num_xapps):
            mal_t = self.is_malicious_matrix[xid]
            if mal_t.any():
                for uid in self.mal_targets[xid]:
                    ue_under_attack[uid, mal_t] = True

        flat_attack_mask = ue_under_attack.flatten()
        n_attack = flat_attack_mask.sum()

        if n_attack > 0:
            ue_ctx.loc[flat_attack_mask, "sinr"] = np.clip(
                ue_ctx.loc[flat_attack_mask, "sinr"] + mc["degradation_sinr"] + self.rng.normal(0, mc["noise_std"], n_attack),
                p["sinr_min"], p["sinr_max"]
            )
            ue_ctx.loc[flat_attack_mask, "throughput"] = np.clip(
                ue_ctx.loc[flat_attack_mask, "throughput"] * mc["degradation_throughput_factor"] * self.rng.normal(1, 0.1, n_attack),
                p["throughput_min"], p["throughput_max"]
            )

        n_dips  = int(self.num_ues * mc["benign_occasional_dip_prob"] * self.num_timesteps)
        dip_idx = self.rng.integers(0, len(ue_ctx), size=n_dips)
        ue_ctx.loc[dip_idx, "sinr"] = np.clip(
            ue_ctx.loc[dip_idx, "sinr"] + self.rng.normal(-3, 1, size=n_dips),
            p["sinr_min"], p["sinr_max"]
        )
        return ue_ctx

    def _assemble(self, actions: pd.DataFrame, ue_ctx: pd.DataFrame) -> pd.DataFrame:
        logger.info("Executing O(1) Context Merge...")
        T       = self.num_timesteps
        actions = actions.copy()

        ctx_idx = (actions["target_ue"] * T + actions["timestamp"]).values
        actions["window_id"]        = actions["timestamp"] // self.window_size
        actions["rsrp"]             = ue_ctx["rsrp"].values[ctx_idx]
        actions["sinr"]             = ue_ctx["sinr"].values[ctx_idx]
        actions["cell_load"]        = ue_ctx["cell_load"].values[ctx_idx]
        actions["ue_count"]         = ue_ctx["ue_count"].values[ctx_idx]
        actions["throughput"]       = ue_ctx["throughput"].values[ctx_idx]
        actions["qos_score"]        = ue_ctx["qos_score"].values[ctx_idx]

        actions = actions.sort_values(["target_ue", "timestamp"]).reset_index(drop=True)
        grp = actions.groupby(["target_ue", "window_id"])
        actions["sinr_delta"]       = actions["sinr"] - grp["sinr"].transform('first')
        actions["throughput_delta"] = actions["throughput"] - grp["throughput"].transform('first')
        actions["qos_delta"]        = actions["qos_score"] - grp["qos_score"].transform('first')

        actions["msg_type"]    = "RIC_CONTROL_REQ"
        actions["routing_key"] = (
            "xapp_" + actions["xapp_id"].astype(str) +
            ".ue_"  + actions["target_ue"].astype(str) +
            ".cell_" + actions["target_cell"].astype(str)
        )
        return actions.sort_values(["timestamp", "xapp_id"]).reset_index(drop=True)


# ==============================================================================
# TELEMETRY GENERATOR
# ==============================================================================
class TelemetryGenerator:
    def __init__(self, config: Dict[str, Any], is_malicious_matrix: np.ndarray):
        self.config    = config
        self.rng       = np.random.default_rng(config["seed"])
        topo           = config["topology"]
        self.num_xapps = topo["num_xapps"]
        self.num_ues   = topo["num_ues"]
        self.num_cells = topo["num_cells"]
        self.num_ts    = config["time"]["num_timesteps"]
        self.cell_ids  = self.rng.integers(0, self.num_cells, size=self.num_xapps)
        self.is_malicious_matrix = is_malicious_matrix

    def generate(self) -> pd.DataFrame:
        logger.info("Generating Telemetry data (Vectorized)...")
        T = self.num_ts
        X = self.num_xapps

        xapp_ids   = np.arange(X).repeat(T)
        timestamps = np.tile(np.arange(T), X)

        load_modes_flat = _get_load_mode_vectorized(timestamps)
        mal      = self.is_malicious_matrix.flatten().astype(int)
        ai       = np.ones(X * T)
        mal_mask = mal == 1
        ai[mal_mask] = self.rng.uniform(1.2, 1.8, size=mal_mask.sum())

        prb_dl  = self.rng.uniform(30, 70, size=X * T)
        prb_ul  = self.rng.uniform(20, 60, size=X * T)
        rmr_rate = self.rng.uniform(5, 15, size=X * T)
        cpu     = self.rng.uniform(20, 60, size=X * T)
        memory  = self.rng.uniform(200, 500, size=X * T)

        prb_dl[mal_mask]   *= ai[mal_mask]
        rmr_rate[mal_mask] *= ai[mal_mask]
        cpu[mal_mask]      *= 1.3
        memory[mal_mask]   *= 1.2

        return pd.DataFrame({
            "timestamp":         timestamps,
            "xapp_id":           xapp_ids,
            "xapp_home_cell":    self.cell_ids[xapp_ids],
            "load_mode":         load_modes_flat,
            "prb_usage_dl":      np.round(prb_dl, 4),
            "prb_usage_ul":      np.round(prb_ul, 4),
            "total_prb_available": 100,
            "ue_count_raw":      self.num_ues,
            "rmr_msg_rate":      np.round(rmr_rate, 4),
            "rmr_msg_size_avg":  np.round(self.rng.uniform(200, 800, size=X * T), 4),
            "cpu_usage":         np.round(cpu, 4),
            "memory_usage":      np.round(memory, 4),
            "execution_latency": np.round(self.rng.uniform(1, 10, size=X * T), 4),
            "is_malicious":      mal,
        })


# ==============================================================================
# DATASET GENERATOR (Orchestrator)
# ==============================================================================
class DatasetGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        T        = config["time"]["num_timesteps"]
        X        = config["topology"]["num_xapps"]
        timeline = config["topology"]["malicious_timeline"]

        rng          = np.random.default_rng(config["seed"])
        t_pct        = np.arange(T) / T
        active_ratio = np.zeros(T)

        for start, end, ratio in timeline:
            mask = (t_pct >= start) & (t_pct <= end)
            active_ratio[mask] = ratio

        is_malicious_matrix  = np.zeros((X, T), dtype=bool)
        compromise_order     = rng.permutation(X)

        for t in range(T):
            k = int(X * active_ratio[t])
            if k > 0:
                is_malicious_matrix[compromise_order[:k], t] = True

        mal_targets = {
            xid: rng.choice(config["topology"]["num_ues"],
                            size=config["malicious"]["target_ue_count"],
                            replace=False).tolist()
            for xid in range(X)
        }

        self.is_malicious_matrix = is_malicious_matrix
        self.mal_targets         = mal_targets
        self.phys_gen = PhysicsGenerator(config, is_malicious_matrix, mal_targets)
        self.tel_gen  = TelemetryGenerator(config, is_malicious_matrix)

    def generate(self) -> pd.DataFrame:
        logger.info("Starting Dataset Generation with Dynamic Timeline...")
        phys_df = self.phys_gen.generate()
        tel_df  = self.tel_gen.generate()

        logger.info("Final O(1) Telemetry Matrix Construction...")
        T       = self.config["time"]["num_timesteps"]
        tel_idx = (phys_df["xapp_id"] * T + phys_df["timestamp"]).values

        for col in tel_df.columns:
            if col not in ["timestamp", "xapp_id"]:
                phys_df[col] = tel_df[col].values[tel_idx]

        cols    = [c for c in phys_df.columns if c != "is_malicious"] + ["is_malicious"]
        dataset = phys_df[cols]
        logger.info(f"Generation Complete. Rows: {len(dataset):,}, Malicious: {dataset['is_malicious'].sum():,}")
        return dataset


# ==============================================================================
# PUBLIC API
# ==============================================================================
def generate_dataset(config: Dict[str, Any] = None) -> pd.DataFrame:
    """Generate a synthetic O-RAN IDS dataset. Returns the DataFrame."""
    if config is None:
        config = DEFAULT_CONFIG
    gen = DatasetGenerator(config)
    return gen.generate()
