"""
topology_alignment_analysis.py

Run alpha-alignment experiments across multiple network topologies and
produce:
1. Replication-level results
2. Topology-wise summary statistics
3. Topology-wise trend statistics for H3
4. Alpha-alignment relationship plots

Main hypothesis (H3):
    As alpha increases, the alignment distance to the KOL decreases.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr

from functions import inject_kol, simulate
from init_belief import InitialBelief
from logic_matrix import LogicMatrix
from metrics import alignment_with_kol
from social_network import SocialNetwork


DEFAULT_MODELS = ("ER", "WS", "BA", "RR")


def _build_inputs(
    agent_num: int,
    topic_num: int,
    topology: str,
    belief_mode: str,
    random_beta: bool,
    seed: int,
):
    network = SocialNetwork(
        n=agent_num,
        model=topology,
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


def _run_single_experiment(
    W: np.ndarray,
    C_tensor: np.ndarray,
    C_base: np.ndarray,
    X0: np.ndarray,
    alpha: float,
    T: int,
    kol_index: int,
):
    C_kol = inject_kol(C_tensor, C_base, alpha=alpha, kol_index=kol_index)
    traj = simulate(W, C_kol, X0, T=T)
    alignment_traj = alignment_with_kol(traj, kol_index=kol_index)
    return {
        "traj": traj,
        "alignment_traj": alignment_traj,
        "final_alignment": float(alignment_traj[-1]),
        "mean_alignment": float(np.mean(alignment_traj)),
    }


def run_topology_alpha_sweep(
    agent_num: int,
    topic_num: int,
    T: int,
    alpha_values,
    topologies=DEFAULT_MODELS,
    n_replications: int = 50,
    belief_mode: str = "uniform",
    random_beta: bool = True,
    kol_index: int = 0,
    base_seed: int = 1942340,
):
    """
    Sweep alpha across multiple topologies and replications.

    Returns a list of dict rows suitable for summary tables and plotting.
    """
    rows = []
    alpha_values = [float(alpha) for alpha in alpha_values]

    for topology in topologies:
        for rep in range(n_replications):
            seed = base_seed + rep
            W, C_tensor, C_base, X0 = _build_inputs(
                agent_num=agent_num,
                topic_num=topic_num,
                topology=topology,
                belief_mode=belief_mode,
                random_beta=random_beta,
                seed=seed,
            )

            for alpha in alpha_values:
                outcome = _run_single_experiment(
                    W=W,
                    C_tensor=C_tensor,
                    C_base=C_base,
                    X0=X0,
                    alpha=alpha,
                    T=T,
                    kol_index=kol_index,
                )
                rows.append(
                    {
                        "topology": topology,
                        "replication": rep,
                        "seed": seed,
                        "alpha": alpha,
                        "final_alignment": outcome["final_alignment"],
                        "mean_alignment": outcome["mean_alignment"],
                    }
                )

    return rows


def summarize_results(rows):
    """
    Aggregate replications by topology and alpha.
    """
    grouped = {}
    for row in rows:
        key = (row["topology"], row["alpha"])
        grouped.setdefault(key, []).append(row["final_alignment"])

    summary_rows = []
    for (topology, alpha), values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        sem = std / np.sqrt(values.size) if values.size > 0 else 0.0
        ci95 = 1.96 * sem
        summary_rows.append(
            {
                "topology": topology,
                "alpha": alpha,
                "n": int(values.size),
                "mean_final_alignment": mean,
                "std_final_alignment": std,
                "ci95_final_alignment": float(ci95),
            }
        )
    return summary_rows


def analyse_h3_by_topology(rows):
    """
    Fit topology-wise alpha -> alignment trends.

    Reports both:
    - linear regression slope
    - Spearman monotonic correlation
    """
    topology_rows = {}
    for row in rows:
        topology_rows.setdefault(row["topology"], []).append(row)

    analysis_rows = []
    for topology, items in sorted(topology_rows.items()):
        alpha = np.asarray([item["alpha"] for item in items], dtype=float)
        alignment = np.asarray([item["final_alignment"] for item in items], dtype=float)

        lin = linregress(alpha, alignment)
        spear = spearmanr(alpha, alignment)

        analysis_rows.append(
            {
                "topology": topology,
                "slope": float(lin.slope),
                "intercept": float(lin.intercept),
                "r_value": float(lin.rvalue),
                "r_squared": float(lin.rvalue ** 2),
                "p_value_linear": float(lin.pvalue),
                "spearman_rho": float(spear.statistic),
                "p_value_spearman": float(spear.pvalue),
                "supports_h3": bool(lin.slope < 0 and spear.statistic < 0),
            }
        )

    return analysis_rows


def plot_alpha_alignment(
    summary_rows,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot mean final alignment vs alpha for each topology with 95% CI bars.
    """
    topology_rows = {}
    for row in summary_rows:
        topology_rows.setdefault(row["topology"], []).append(row)

    plt.figure(figsize=(9, 6))

    for topology, items in sorted(topology_rows.items()):
        items = sorted(items, key=lambda x: x["alpha"])
        alpha = [item["alpha"] for item in items]
        mean_alignment = [item["mean_final_alignment"] for item in items]
        ci95 = [item["ci95_final_alignment"] for item in items]

        plt.errorbar(
            alpha,
            mean_alignment,
            yerr=ci95,
            marker="o",
            linewidth=2,
            capsize=4,
            label=topology,
        )

    plt.xlabel("KOL influence strength alpha")
    plt.ylabel("Final alignment to KOL")
    plt.title("Alpha-alignment relationship across network topologies")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_topology_panels(
    summary_rows,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot one alpha-alignment panel per topology.
    """
    topology_rows = {}
    for row in summary_rows:
        topology_rows.setdefault(row["topology"], []).append(row)

    topologies = sorted(topology_rows.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, topology in zip(axes, topologies):
        items = sorted(topology_rows[topology], key=lambda x: x["alpha"])
        alpha = [item["alpha"] for item in items]
        mean_alignment = [item["mean_final_alignment"] for item in items]
        ci95 = [item["ci95_final_alignment"] for item in items]

        ax.errorbar(
            alpha,
            mean_alignment,
            yerr=ci95,
            marker="o",
            linewidth=2,
            capsize=4,
        )
        ax.set_title(topology)
        ax.grid(alpha=0.3)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Final alignment")

    for idx in range(len(topologies), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Topology-wise alpha-alignment relationships", y=0.98)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _write_csv(rows, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run alpha-alignment analysis across topologies."
    )
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--topics", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1942340)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--belief-mode", default="uniform")
    parser.add_argument("--kol-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="code/outputs/topology_alignment",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_topology_alpha_sweep(
        agent_num=args.agents,
        topic_num=args.topics,
        T=args.steps,
        alpha_values=args.alphas,
        topologies=args.topologies,
        n_replications=args.replications,
        belief_mode=args.belief_mode,
        random_beta=True,
        kol_index=args.kol_index,
        base_seed=args.seed,
    )
    summary_rows = summarize_results(rows)
    trend_rows = analyse_h3_by_topology(rows)

    _write_csv(rows, output_dir / "replication_results.csv")
    _write_csv(summary_rows, output_dir / "summary_results.csv")
    _write_csv(trend_rows, output_dir / "topology_h3_tests.csv")

    plot_alpha_alignment(
        summary_rows,
        output_path=output_dir / "alpha_alignment_all_topologies.png",
        show=not args.no_show,
    )
    plot_topology_panels(
        summary_rows,
        output_path=output_dir / "alpha_alignment_topology_panels.png",
        show=not args.no_show,
    )

    print(f"Saved outputs to: {output_dir}")
    print("H3 trend summary:")
    for row in trend_rows:
        print(
            f"{row['topology']}: slope={row['slope']:.4f}, "
            f"spearman_rho={row['spearman_rho']:.4f}, "
            f"supports_h3={row['supports_h3']}"
        )


if __name__ == "__main__":
    main()
