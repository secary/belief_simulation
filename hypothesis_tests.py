"""
hypothesis_tests.py

Utilities for simulation-based hypothesis testing in the belief dynamics
framework.

Current main hypothesis:
    Increasing alpha leads to a smaller alignment distance to the KOL.

Recommended workflow:
1. Re-generate trajectories across many seeds.
2. Compare a lower alpha and a higher alpha under paired simulations.
3. Use a paired permutation test on the metric differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions import inject_kol, simulate
from init_belief import InitialBelief
from logic_matrix import LogicMatrix
from metrics import compute_all_metrics
from social_network import SocialNetwork


MetricLike = str


def format_p_value(p_value: float) -> str:
    """
    Format p-values for display without collapsing tiny values to 0.
    """
    if p_value < 1e-16:
        return "< 1e-16"
    return f"{p_value:.3e}"


@dataclass
class PairedTestResult:
    metric_name: str
    alternative: str
    n_pairs: int
    baseline_mean: float
    treatment_mean: float
    observed_effect: float
    p_value: float
    ci_low: float
    ci_high: float
    baseline_values: np.ndarray
    treatment_values: np.ndarray
    differences: np.ndarray

    def summary(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "alternative": self.alternative,
            "n_pairs": self.n_pairs,
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "observed_effect": self.observed_effect,
            "p_value": self.p_value,
            "p_value_display": format_p_value(self.p_value),
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def _metric_from_traj(traj, metric_name: MetricLike, kol_index: int | None = None) -> float:
    metrics = compute_all_metrics(traj, kol_index=kol_index)

    if metric_name not in metrics:
        raise KeyError(f"Unknown metric '{metric_name}'. Available keys: {list(metrics.keys())}")

    value = metrics[metric_name]
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError(
            f"Metric '{metric_name}' is not scalar. "
            "Choose a scalar metric such as 'final_variance' or 'final_alignment'."
        )
    return float(value)


def paired_permutation_test(
    baseline_values,
    treatment_values,
    alternative: str = "two-sided",
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict:
    """
    Paired sign-flip permutation test for treatment - baseline.

    alternative:
        - "two-sided": effect != 0
        - "greater": effect > 0
        - "less": effect < 0
    """
    baseline_values = np.asarray(baseline_values, dtype=float)
    treatment_values = np.asarray(treatment_values, dtype=float)

    if baseline_values.shape != treatment_values.shape:
        raise ValueError("baseline_values and treatment_values must have the same shape.")

    if baseline_values.ndim != 1:
        raise ValueError("baseline_values and treatment_values must be 1D arrays.")

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of {'two-sided', 'greater', 'less'}.")

    rng = np.random.default_rng(seed)
    differences = treatment_values - baseline_values
    observed = float(differences.mean())

    signs = rng.choice([-1.0, 1.0], size=(n_resamples, differences.size))
    null_stats = (signs * differences).mean(axis=1)

    if alternative == "greater":
        exceedances = np.count_nonzero(null_stats >= observed)
    elif alternative == "less":
        exceedances = np.count_nonzero(null_stats <= observed)
    else:
        exceedances = np.count_nonzero(np.abs(null_stats) >= abs(observed))

    # Use the standard +1 correction so Monte Carlo permutation p-values
    # never collapse to exactly 0 just because no resample exceeded the test statistic.
    p_value = (exceedances + 1.0) / (n_resamples + 1.0)

    bootstrap_idx = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    bootstrap_stats = differences[bootstrap_idx].mean(axis=1)
    alpha = 1.0 - ci_level
    ci_low, ci_high = np.quantile(bootstrap_stats, [alpha / 2, 1.0 - alpha / 2])

    return {
        "observed_effect": observed,
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "differences": differences,
    }


def _build_simulation_inputs(
    agent_num: int,
    topic_num: int,
    network_model: str | None,
    belief_mode: str,
    random_beta: bool,
    seed: int,
):
    network = SocialNetwork(
        n=agent_num,
        model=network_model,
        random_beta=random_beta,
        seed=seed,
    ).generate()

    logic = LogicMatrix(
        n=agent_num,
        m=topic_num,
        random_beta=random_beta,
        seed=seed,
    ).generate()

    beliefs = InitialBelief(
        n=agent_num,
        m=topic_num,
        mode=belief_mode,
        seed=seed,
    ).generate()

    return network.W, np.asarray(logic.C_list), logic.C_base, beliefs.X0


def run_alpha_alignment_test(
    agent_num: int,
    topic_num: int,
    T: int,
    alpha_low: float,
    alpha_high: float,
    metric_name: MetricLike = "final_alignment",
    n_replications: int = 200,
    alternative: str = "less",
    network_model: str | None = None,
    belief_mode: str = "uniform",
    random_beta: bool = True,
    kol_index: int = 0,
    base_seed: int = 1942340,
    n_resamples: int = 10000,
) -> PairedTestResult:
    """
    Test whether a larger alpha leads to a smaller alignment distance.

    Default hypothesis for metric='final_alignment':
        H0: E[alignment(alpha_high) - alignment(alpha_low)] = 0
        H1: E[alignment(alpha_high) - alignment(alpha_low)] < 0
    """
    if alpha_high <= alpha_low:
        raise ValueError("alpha_high must be greater than alpha_low.")

    low_alpha_values = []
    high_alpha_values = []

    for rep in range(n_replications):
        seed = base_seed + rep
        W, C_tensor, C_base, X0 = _build_simulation_inputs(
            agent_num=agent_num,
            topic_num=topic_num,
            network_model=network_model,
            belief_mode=belief_mode,
            random_beta=random_beta,
            seed=seed,
        )

        low_alpha_tensor = inject_kol(C_tensor, C_base, alpha=alpha_low, kol_index=kol_index)
        high_alpha_tensor = inject_kol(C_tensor, C_base, alpha=alpha_high, kol_index=kol_index)

        low_alpha_traj = simulate(W, low_alpha_tensor, X0, T=T)
        high_alpha_traj = simulate(W, high_alpha_tensor, X0, T=T)

        low_alpha_values.append(_metric_from_traj(low_alpha_traj, metric_name, kol_index=kol_index))
        high_alpha_values.append(_metric_from_traj(high_alpha_traj, metric_name, kol_index=kol_index))

    low_alpha_values = np.asarray(low_alpha_values, dtype=float)
    high_alpha_values = np.asarray(high_alpha_values, dtype=float)
    test_result = paired_permutation_test(
        baseline_values=low_alpha_values,
        treatment_values=high_alpha_values,
        alternative=alternative,
        n_resamples=n_resamples,
        seed=base_seed,
    )

    return PairedTestResult(
        metric_name=metric_name,
        alternative=alternative,
        n_pairs=n_replications,
        baseline_mean=float(low_alpha_values.mean()),
        treatment_mean=float(high_alpha_values.mean()),
        observed_effect=float(test_result["observed_effect"]),
        p_value=float(test_result["p_value"]),
        ci_low=float(test_result["ci_low"]),
        ci_high=float(test_result["ci_high"]),
        baseline_values=low_alpha_values,
        treatment_values=high_alpha_values,
        differences=np.asarray(test_result["differences"], dtype=float),
    )


if __name__ == "__main__":
    result = run_alpha_alignment_test(
        agent_num=30,
        topic_num=3,
        T=20,
        alpha_low=0.3,
        alpha_high=0.9,
        metric_name="final_alignment",
        n_replications=100,
        alternative="less",
        network_model=None,
        belief_mode="uniform",
        random_beta=True,
        kol_index=0,
        base_seed=1942340,
        n_resamples=5000,
    )
    print(result.summary())
