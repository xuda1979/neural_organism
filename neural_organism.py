import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Utility Functions ---

def softmax(z):
    zmax = np.max(z)
    e = np.exp(z - zmax)
    return e / np.sum(e)

# --- Environment Classes ---

class MixtureDriftGaussians:
    """2-D, K-class classification with drifting Gaussian mixtures."""
    def __init__(self, d=2, k=3, n_components=2, sigma=0.4, range_lim=3.0,
                 steps_per_phase=2000, num_phases=6, seed=0):
        self.d = d; self.k = k; self.n_components = n_components
        self.sigma = sigma; self.range_lim = range_lim
        self.steps_per_phase = steps_per_phase; self.num_phases = num_phases
        self.total_steps = steps_per_phase * num_phases
        self.cur_phase = 0
        self.rng = np.random.RandomState(seed)
        self.centers = self._sample_new_centers()

    def _sample_new_centers(self):
        return self.rng.uniform(-self.range_lim, self.range_lim,
                                size=(self.k, self.n_components, self.d))

    def reset_phase(self, phase_idx):
        self.cur_phase = phase_idx
        self.centers = self._sample_new_centers()

    def sample(self, t):
        phase_idx = t // self.steps_per_phase
        if phase_idx != self.cur_phase:
            self.reset_phase(phase_idx)
        y = self.rng.randint(0, self.k)
        comp = self.rng.randint(0, self.n_components)
        mean = self.centers[y, comp]
        x = mean + self.rng.normal(0, self.sigma, size=(self.d,))
        return x.astype(np.float64), int(y)

class ContextualBanditDrift:
    """Non-stationary contextual bandit in R^2 with drifting hotspots."""
    def __init__(self, d=2, A=3, n_components=2, sigma=0.5, range_lim=3.0,
                 steps_per_phase=2000, num_phases=6, seed=0):
        self.d = d; self.A = A; self.n_components = n_components
        self.sigma = sigma; self.range_lim = range_lim
        self.steps_per_phase = steps_per_phase; self.num_phases = num_phases
        self.total_steps = steps_per_phase * num_phases
        self.cur_phase = 0
        self.rng = np.random.RandomState(seed)
        self.centers = self._new_centers()
        self.scales  = self._new_scales()

    def _new_centers(self):
        return self.rng.uniform(-self.range_lim, self.range_lim,
                                size=(self.A, self.n_components, self.d))

    def _new_scales(self):
        return 0.6 + 0.8 * self.rng.rand(self.A, self.n_components)

    def reset_phase(self, phase_idx):
        self.cur_phase = phase_idx
        self.centers = self._new_centers()
        self.scales  = self._new_scales()

    def _reward_prob(self, x, a):
        diffs = self.centers[a] - x
        dist2 = np.sum(diffs * diffs, axis=1)
        logits = -dist2 / (2.0 * (self.sigma ** 2)) * self.scales[a]
        val = np.log1p(np.sum(np.exp(logits)))
        p = 1.0 / (1.0 + np.exp(-val))
        return float(p)

    def sample(self, t):
        phase_idx = t // self.steps_per_phase
        if phase_idx != self.cur_phase:
            self.reset_phase(phase_idx)
        x = self.rng.uniform(-self.range_lim, self.range_lim, size=(self.d,))
        return x.astype(np.float64)

    def pull(self, x, a):
        p = self._reward_prob(x, a)
        r = 1 if self.rng.rand() < p else 0
        return r, p

# --- Baseline Models ---

@dataclass
class LogisticSGD:
    d: int; k: int; lr: float = 0.05; wd: float = 1e-4
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    name: str = "LogReg"

    def __post_init__(self):
        rng = np.random.RandomState(0)
        self.W = rng.randn(self.d, self.k) * 0.01
        self.b = np.zeros(self.k)

    def predict(self, x):
        scores = x @ self.W + self.b
        p = softmax(scores)
        return int(np.argmax(p))

    def update(self, x, y):
        scores = x @ self.W + self.b
        p = softmax(scores)
        y_one = np.zeros_like(p); y_one[y] = 1.0
        grad_scores = p - y_one
        gW = np.outer(x, grad_scores) + self.wd * self.W
        gb = grad_scores
        self.W -= self.lr * gW
        self.b -= self.lr * gb

@dataclass
class MLPOnline:
    d: int; k: int; h: int = 32; lr: float = 0.01; wd: float = 1e-5
    name: str = "MLP(32)"
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.RandomState(0)
        limit1 = np.sqrt(6 / (self.d + self.h))
        limit2 = np.sqrt(6 / (self.h + self.k))
        self.W1 = rng.uniform(-limit1, limit1, size=(self.d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = rng.uniform(-limit2, limit2, size=(self.h, self.k))
        self.b2 = np.zeros(self.k)

    def _forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        scores = h @ self.W2 + self.b2
        p = softmax(scores)
        return h, p

    def predict(self, x):
        _, p = self._forward(x)
        return int(np.argmax(p))

    def update(self, x, y):
        h, p = self._forward(x)
        y_one = np.zeros_like(p); y_one[y] = 1.0
        dscores = p - y_one
        gW2 = np.outer(h, dscores) + self.wd * self.W2
        gb2 = dscores
        dh = self.W2 @ dscores
        dz1 = (1.0 - h * h) * dh
        gW1 = np.outer(x, dz1) + self.wd * self.W1
        gb1 = dz1
        self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
        self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1

class LinUCB:
    """Standard LinUCB for contextual bandits."""
    def __init__(self, d, A, alpha=0.8, lam=1.0, name: str | None = None):
        self.d = d + 1  # bias
        self.A = A
        self.alpha = alpha
        self.lam = lam
        self.As = [lam * np.eye(self.d) for _ in range(A)]
        self.bs = [np.zeros(self.d) for _ in range(A)]
        self.name = name if name is not None else f"LinUCB(alpha={alpha})"

    def _feat(self, x):
        return np.concatenate([x, [1.0]])

    def select(self, x):
        phi = self._feat(x)
        best_a, best_u = 0, -1e9
        for a in range(self.A):
            A_inv = np.linalg.inv(self.As[a])
            theta = A_inv @ self.bs[a]
            mu = theta @ phi
            ucb = mu + self.alpha * np.sqrt(phi @ A_inv @ phi)
            if ucb > best_u:
                best_u, best_a = ucb, a
        return best_a

    def update(self, x, a, r):
        phi = self._feat(x)
        self.As[a] += np.outer(phi, phi)
        self.bs[a] += r * phi

# --- Runtime Gate and Growing RBF Network ---

class RuntimeGate:
    """A simple runtime-enforced gate based on a consumable budget."""
    def __init__(self, budget: int | None):
        self._budget = budget

    def is_open(self) -> bool:
        if self._budget is None:
            return True
        return self._budget > 0

    def consume(self):
        if self._budget is not None:
            self._budget -= 1

class GrowingRBFNetPlastic:
    """A Growing RBF Network with plastic synapses and runtime budgeting."""
    def __init__(self, d, k, sigma=0.65, lr_w=0.25, lr_c=0.06, wd=1e-5,
                 min_phi_to_cover=0.25, usage_decay=0.995, max_prototypes=90,
                 growth_gate: RuntimeGate | None = None,
                 hebb_eta=0.5, hebb_decay=0.95, gate_beta=1.0, gate_top=16, key_lr=0.02,
                 r_merge=0.4, merge_every=500, name="GRBFN+Plastic"):
        self.d = d; self.k = k; self.sigma = sigma
        self.lr_w = lr_w; self.lr_c = lr_c; self.wd = wd
        self.min_phi_to_cover = min_phi_to_cover
        self.usage_decay = usage_decay; self.max_prototypes = max_prototypes; self.name = name
        self.growth_gate = growth_gate if growth_gate is not None else RuntimeGate(None)
        self.hebb_eta = hebb_eta; self.hebb_decay = hebb_decay
        self.gate_beta = gate_beta; self.gate_top = gate_top; self.key_lr = key_lr
        self.r_merge = r_merge; self.merge_every = merge_every
        self.t = 0

        rng = np.random.RandomState(0)
        self.centers = rng.randn(k, d) * 0.01
        self.W = np.zeros((k, k))
        for c in range(k):
            self.W[c, c] = 1.0
        self.H = np.zeros((k, k))
        self.K = rng.randn(k, d) * 0.01
        self.usage = np.ones(k)
        self.M = k

    def _phi(self, x):
        diffs = self.centers - x
        dist2 = np.sum(diffs * diffs, axis=1)
        phi = np.exp(-dist2 / (2.0 * (self.sigma ** 2)))
        return phi, dist2

    def _gate(self, x, phi):
        g = 1.0 / (1.0 + np.exp(-self.gate_beta * (self.K @ x)))
        s = phi * g
        if self.M <= self.gate_top:
            mask = np.ones(self.M, dtype=bool)
        else:
            idx = np.argpartition(s, -self.gate_top)[-self.gate_top:]
            mask = np.zeros(self.M, dtype=bool); mask[idx] = True
        return s, mask

    def predict(self, x, return_proba=False):
        phi, _ = self._phi(x)
        s, mask = self._gate(x, phi)
        scores = (phi * mask) @ (self.W + self.H)
        zmax = np.max(scores); p = np.exp(scores - zmax); p /= np.sum(p)
        pred = int(np.argmax(p))
        if return_proba:
            return pred, p
        return pred

    def _spawn(self, x, y):
        if not self.growth_gate.is_open():
            return
        if self.M >= self.max_prototypes:
            self._prune()
            if self.M >= self.max_prototypes:
                return
        self.centers = np.vstack([self.centers, x.reshape(1,-1)])
        new_w = np.zeros((1, self.k)); new_h = np.zeros((1, self.k))
        new_w[0, y] = 1.0
        self.W = np.vstack([self.W, new_w])
        self.H = np.vstack([self.H, new_h])
        self.K = np.vstack([self.K, x.reshape(1,-1)])
        self.usage = np.concatenate([self.usage, np.array([1.0])])
        self.M += 1
        self.growth_gate.consume()

    def _prune(self):
        if self.M <= self.k:
            return
        idx = int(np.argmin(self.usage))
        self.centers = np.delete(self.centers, idx, axis=0)
        self.W = np.delete(self.W, idx, axis=0)
        self.H = np.delete(self.H, idx, axis=0)
        self.K = np.delete(self.K, idx, axis=0)
        self.usage = np.delete(self.usage, idx, axis=0)
        self.M -= 1

    def _maybe_merge(self):
        if self.t % self.merge_every != 0 or self.M <= self.k:
            return
        keep = np.ones(self.M, dtype=bool)
        for i in range(self.M):
            if not keep[i]:
                continue
            for j in range(i + 1, self.M):
                if not keep[j]:
                    continue
                if np.sum((self.centers[i] - self.centers[j]) ** 2) < self.r_merge ** 2:
                    self.centers[i] = 0.5 * (self.centers[i] + self.centers[j])
                    self.W[i] += self.W[j]
                    self.H[i] += self.H[j]
                    self.K[i] = 0.5 * (self.K[i] + self.K[j])
                    keep[j] = False
        if np.any(~keep):
            self.centers = self.centers[keep]
            self.W = self.W[keep]
            self.H = self.H[keep]
            self.K = self.K[keep]
            self.usage = self.usage[keep]
            self.M = self.centers.shape[0]

    def _update_weights_and_centers(self, x, y, phi, mask):
        scores = (phi * mask) @ (self.W + self.H)
        zmax = np.max(scores); exp_scores = np.exp(scores - zmax)
        p = exp_scores / np.sum(exp_scores)
        y_one = np.zeros(self.k); y_one[y] = 1.0
        ds = p - y_one
        gW = np.outer(phi * mask, ds) + self.wd * self.W
        self.W -= self.lr_w * gW
        dphi = ((self.W + self.H) @ ds) * mask
        dcenters = (phi[:, None] * (x - self.centers) / (self.sigma ** 2)) * dphi[:, None]
        self.centers -= self.lr_c * dcenters
        return p, ds

    def _update_plasticity(self, x, y, phi, mask, p, ds):
        y_one = np.zeros(self.k); y_one[y] = 1.0
        self.H *= self.hebb_decay
        self.H += self.hebb_eta * np.outer(phi * mask, (y_one - p))
        corr = -np.sum(ds)
        self.K += self.key_lr * ((phi * mask)[:, None] * (x - self.K)) * corr

    def _perform_structural_maintenance(self, s):
        self.usage += s * 0.05
        if self.M > self.max_prototypes:
            self._prune()
        self._maybe_merge()

    def update(self, x, y):
        self.t += 1
        self.usage *= self.usage_decay
        phi, dist2 = self._phi(x)
        s, mask = self._gate(x, phi)
        if np.max(s) < self.min_phi_to_cover:
            self._spawn(x, y)
            phi, dist2 = self._phi(x)
            s, mask = self._gate(x, phi)
        p, ds = self._update_weights_and_centers(x, y, phi, mask)
        self._update_plasticity(x, y, phi, mask, p, ds)
        self._perform_structural_maintenance(s)

# --- Experiment Utilities ---

plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FIG_SIZE_BANDIT = (8, 5)
FIG_SIZE_CLASSIFICATION = (10, 6)

def plot_bandit_results(results: dict, out_dir: str, plot_settings: dict):
    filename_prefix = plot_settings['filename_prefix']
    T = len(next(iter(results.values()))['cum_rewards'])
    x = np.arange(T)
    steps_per_phase = T // 6

    plt.figure(figsize=FIG_SIZE_BANDIT)
    for name, data in results.items():
        plt.plot(x, data['cum_rewards'], label=name)
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray')
    plt.title("Cumulative Reward Under Drift", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Cumulative Reward", fontsize=FONT_SIZE_LABEL)
    plt.legend()
    cumr_path = os.path.join(out_dir, f"{filename_prefix}_cumreward.png")
    plt.tight_layout(); plt.savefig(cumr_path); plt.close()

    plt.figure(figsize=FIG_SIZE_BANDIT)
    for name, data in results.items():
        plt.plot(x, data['cum_regrets'], label=name)
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray')
    plt.title("Cumulative Regret Under Drift", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Cumulative Regret", fontsize=FONT_SIZE_LABEL)
    plt.legend()
    regret_path = os.path.join(out_dir, f"{filename_prefix}_cumregret.png")
    plt.tight_layout(); plt.savefig(regret_path); plt.close()

    return cumr_path, regret_path

def save_bandit_summary(summary_data: list, out_dir: str, filename: str):
    summary_df = pd.DataFrame(summary_data).sort_values("Policy")
    summary_path = os.path.join(out_dir, filename)
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    return summary_path

def plot_classification_results(results: dict, out_dir: str, plot_settings: dict, window_size: int = 100):
    filename = plot_settings['acc_out_file']
    plt.figure(figsize=FIG_SIZE_CLASSIFICATION)
    for name, corrects in results.items():
        smoothed_acc = pd.Series(corrects).rolling(window=window_size, min_periods=1).mean()
        plt.plot(smoothed_acc, label=name)
    T = len(next(iter(results.values())))
    steps_per_phase = T // 6
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray', alpha=0.7)
    plt.title(f"Sliding Window Accuracy (size={window_size})", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE_LABEL)
    plt.ylim(0, 1.05)
    plt.legend()
    acc_path = os.path.join(out_dir, filename)
    plt.tight_layout(); plt.savefig(acc_path); plt.close()
    return acc_path

def plot_prototypes(counts: np.ndarray, model_name: str, out_dir: str, filename: str):
    plt.figure(figsize=(8, 3.8))
    plt.plot(counts, label=model_name)
    steps_per_phase = len(counts) // 6
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray')
    plt.title("Prototype Growth Over Time", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("# Prototypes", fontsize=FONT_SIZE_LABEL)
    plt.legend()
    proto_path = os.path.join(out_dir, filename)
    plt.tight_layout(); plt.savefig(proto_path); plt.close()
    return proto_path

# --- Experiment Configurations ---

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    name: str
    seed: int
    steps_per_phase: int
    num_phases: int
    env: EnvConfig
    models: List[ModelConfig]
    plot_settings: Dict[str, Any] = field(default_factory=dict)

BASE_ENV_PARAMS = {
    "d": 2,
    "steps_per_phase": 2000,
    "num_phases": 6,
}

BASE_GRBF_PARAMS = {
    "sigma": 0.7, "lr_w": 0.25, "lr_c": 0.06, "min_phi_to_cover": 0.25,
    "max_prototypes": 90, "hebb_eta": 0.6, "hebb_decay": 0.94,
    "gate_beta": 1.2, "gate_top": 16, "key_lr": 0.03, "r_merge": 0.4,
    "merge_every": 600,
}

supervised_drift_config = ExperimentConfig(
    name="supervised_drift",
    seed=21,
    steps_per_phase=BASE_ENV_PARAMS["steps_per_phase"],
    num_phases=BASE_ENV_PARAMS["num_phases"],
    env=EnvConfig(
        name="MixtureDriftGaussians",
        params={"d": 2, "k": 3, "n_components": 2, "sigma": 0.4},
    ),
    models=[
        ModelConfig(name="LogisticSGD", params={"d": 2, "k": 3, "lr": 0.05, "wd": 1e-4}),
        ModelConfig(name="MLPOnline", params={"d": 2, "k": 3, "h": 32, "lr": 0.01, "wd": 1e-5}),
        ModelConfig(name="GRBFN+Plastic", params={
            **BASE_GRBF_PARAMS,
            "d": 2, "k": 3, "sigma": 0.65, "r_merge": 0.35, "growth_gate_budget": None,
        }),
    ],
    plot_settings={
        "type": "classification",
        "proto_out_file": "classification_prototypes.png",
        "acc_out_file": "classification_accuracy.png",
    }
)

bandit_drift_config = ExperimentConfig(
    name="bandit_drift",
    seed=33,
    steps_per_phase=BASE_ENV_PARAMS["steps_per_phase"],
    num_phases=BASE_ENV_PARAMS["num_phases"],
    env=EnvConfig(
        name="ContextualBanditDrift",
        params={"d": 2, "A": 3, "n_components": 2, "sigma": 0.5},
    ),
    models=[
        ModelConfig(name="LinUCB", params={"d": 2, "A": 3, "alpha": 0.8, "lam": 1.0}),
        ModelConfig(name="GRBFN+Plastic", params={
            **BASE_GRBF_PARAMS,
            "d": 2, "k": 3, "growth_gate_budget": 22,
        }),
    ],
    plot_settings={
        "type": "bandit",
        "filename_prefix": "bandit",
    }
)

bandit_drift_plusmlp_config = ExperimentConfig(
    name="bandit_drift_plusmlp",
    seed=37,
    steps_per_phase=BASE_ENV_PARAMS["steps_per_phase"],
    num_phases=BASE_ENV_PARAMS["num_phases"],
    env=EnvConfig(
        name="ContextualBanditDrift",
        params={"d": 2, "A": 3, "n_components": 2, "sigma": 0.5},
    ),
    models=[
        ModelConfig(name="LinUCB", params={"d": 2, "A": 3, "alpha": 0.8, "lam": 1.0}),
        ModelConfig(name="GRBFN+Plastic", params={
            **BASE_GRBF_PARAMS,
            "d": 2, "k": 3, "growth_gate_budget": 22,
        }),
        ModelConfig(name="MLPOnline", params={"d": 2, "k": 3, "h": 32, "lr": 0.01, "wd": 1e-5}),
    ],
    plot_settings={
        "type": "bandit",
        "filename_prefix": "bandit_plusmlp",
    }
)

CONFIGS = {
    "supervised": supervised_drift_config,
    "bandit": bandit_drift_config,
    "bandit_plusmlp": bandit_drift_plusmlp_config,
}

# --- Runner Helpers ---

def get_env(config: ExperimentConfig):
    env_map = {
        "MixtureDriftGaussians": MixtureDriftGaussians,
        "ContextualBanditDrift": ContextualBanditDrift,
    }
    env_class = env_map[config.env.name]
    env_params = {**config.env.params, "seed": config.seed}
    return env_class(**env_params)

def get_models(config: ExperimentConfig):
    model_map = {
        "LogisticSGD": LogisticSGD,
        "MLPOnline": MLPOnline,
        "LinUCB": LinUCB,
        "GRBFN+Plastic": GrowingRBFNetPlastic,
    }
    models = []
    for model_conf in config.models:
        model_class = model_map[model_conf.name]
        params = model_conf.params.copy()
        if model_conf.name == "GRBFN+Plastic":
            budget = params.pop("growth_gate_budget", None)
            if config.env.name == "ContextualBanditDrift":
                params["k"] = params.pop("A", params.get("k"))
            params["growth_gate"] = RuntimeGate(budget=budget)
        if config.env.name == "ContextualBanditDrift" and "k" in params:
            params["k"] = params.pop("A", params.get("k"))
        models.append(model_class(name=model_conf.name, **params))
    return models

def run_supervised(config: ExperimentConfig, out_dir: str):
    env = get_env(config)
    models = get_models(config)
    total_steps = config.steps_per_phase * config.num_phases
    corrects = {m.name: np.zeros(total_steps, dtype=float) for m in models}
    proto_counts = {
        m.name: np.zeros(total_steps, dtype=int)
        for m in models if isinstance(m, GrowingRBFNetPlastic)
    }
    for t in range(total_steps):
        x, y = env.sample(t)
        for m in models:
            yhat = m.predict(x)
            corrects[m.name][t] = 1.0 if yhat == y else 0.0
            m.update(x, y)
            if isinstance(m, GrowingRBFNetPlastic):
                proto_counts[m.name][t] = m.M
    print("--- Supervised Experiment Results ---")
    overall_df = pd.DataFrame({
        "Model": [m.name for m in models],
        "Overall Accuracy": [round(np.mean(corrects[m.name]), 4) for m in models],
    }).sort_values("Model")
    print("\nOverall Performance:")
    print(overall_df)
    os.makedirs(out_dir, exist_ok=True)
    overall_df.to_csv(os.path.join(out_dir, "classification_overall.csv"), index=False)
    drift_points = [i * config.steps_per_phase for i in range(1, config.num_phases)]
    post200 = {m.name: [] for m in models}
    post500 = {m.name: [] for m in models}
    for dp in drift_points:
        for m in models:
            post200[m.name].append(np.mean(corrects[m.name][dp : dp + 200]))
            post500[m.name].append(np.mean(corrects[m.name][dp : dp + 500]))
    post_df = pd.DataFrame({
        "Model": [m.name for m in models],
        "Post@200(avg)": [round(np.mean(v), 4) for v in post200.values()],
        "Post@500(avg)": [round(np.mean(v), 4) for v in post500.values()],
    }).sort_values("Model")
    print("\nPost-Drift Adaptation:")
    print(post_df)
    post_df.to_csv(os.path.join(out_dir, "classification_postdrift.csv"), index=False)
    plot_settings_for_func = config.plot_settings.copy()
    plot_settings_for_func.pop("type", None)
    plot_classification_results(corrects, out_dir, plot_settings=plot_settings_for_func)
    for model_name, counts in proto_counts.items():
        plot_prototypes(counts, model_name, out_dir, config.plot_settings["proto_out_file"])
    print(f"\nPlots saved to {out_dir}")

def run_bandit(config: ExperimentConfig, out_dir: str):
    env = get_env(config)
    models = get_models(config)
    A = config.env.params["A"]
    total_steps = config.steps_per_phase * config.num_phases
    results = {m.name: {"cum_rewards": np.zeros(total_steps), "cum_regrets": np.zeros(total_steps)} for m in models}
    proto_counts = {
        m.name: np.zeros(total_steps, dtype=int)
        for m in models if isinstance(m, GrowingRBFNetPlastic)
    }
    for t in range(total_steps):
        x = env.sample(t)
        ps = [env._reward_prob(x, a) for a in range(A)]
        best_r_prob = np.max(ps)
        for m in models:
            if isinstance(m, LinUCB):
                a = m.select(x)
                r, _ = env.pull(x, a)
                m.update(x, a, r)
            elif isinstance(m, (GrowingRBFNetPlastic, MLPOnline)):
                if isinstance(m, GrowingRBFNetPlastic):
                    a, p = m.predict(x, return_proba=True)
                else:
                    _, p = m._forward(x)
                    a = int(np.argmax(p))
                r, _ = env.pull(x, a)
                if r > 0:
                    y = a
                else:
                    y = int(np.argsort(p)[-2])
                m.update(x, y)
            else:
                raise TypeError(f"Model {m.name} not supported in bandit experiment.")
            regret = best_r_prob - ps[a]
            results[m.name]["cum_rewards"][t] = r if t == 0 else results[m.name]["cum_rewards"][t-1] + r
            results[m.name]["cum_regrets"][t] = regret if t == 0 else results[m.name]["cum_regrets"][t-1] + regret
            if isinstance(m, GrowingRBFNetPlastic):
                proto_counts[m.name][t] = m.M
    print("--- Bandit Experiment Results ---")
    summary_data = []
    for m in models:
        final_protos = np.nan
        budget_rem = np.nan
        if isinstance(m, GrowingRBFNetPlastic):
            final_protos = proto_counts[m.name][-1]
            budget_rem = m.growth_gate._budget
        summary_data.append({
            "Policy": m.name,
            "Final Cumulative Reward": results[m.name]["cum_rewards"][-1],
            "Final Cumulative Regret": results[m.name]["cum_regrets"][-1],
            "Final #Prototypes": final_protos,
            "Growth Budget Remaining": budget_rem,
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    os.makedirs(out_dir, exist_ok=True)
    prefix = config.plot_settings["filename_prefix"]
    summary_path = save_bandit_summary(summary_data, out_dir, f"{prefix}_summary.csv")
    print(f"\nSummary saved to {summary_path}")
    plot_settings_for_func = config.plot_settings.copy()
    plot_settings_for_func.pop("type", None)
    plot_bandit_results(results, out_dir, plot_settings=plot_settings_for_func)
    print(f"Plots saved to {out_dir}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Run experiments for the Neural Organism project.")
    parser.add_argument(
        "experiment",
        type=str,
        choices=CONFIGS.keys(),
        help="The name of the experiment to run.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    args = parser.parse_args()
    config = CONFIGS[args.experiment]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Running experiment: {config.name} (Seed: {config.seed})")
    if config.plot_settings["type"] == "classification":
        run_supervised(config, args.out_dir)
    elif config.plot_settings["type"] == "bandit":
        run_bandit(config, args.out_dir)
    else:
        raise ValueError(f"Unknown experiment type: {config.plot_settings['type']}")

if __name__ == "__main__":
    main()
