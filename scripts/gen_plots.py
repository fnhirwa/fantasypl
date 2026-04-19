r"""Generate all FPLX backtest plots.

Reads the two backtest JSON files and produces seven figures:
  fig_inference_ablation.{pdf,png}   -- MSE ablation, both seasons
  fig_inference_mae.{pdf,png}        -- MAE ablation, both seasons
  fig_calibration.{pdf,png}          -- Coverage at 50% and 95% CI
  fig_optimization_strategies.{pdf,png} -- Mean pts/GW per strategy
  fig_risk_tradeoff.{pdf,png}        -- Mean pts vs CV as lambda varies
  fig_lagrangian_vs_ilp.{pdf,png}    -- Total pts: ILP vs Lagrangian vs Oracle
  fig_summary_table.{pdf,png}        -- Strategy summary table

Usage
-----
  python scripts/gen_plots.py \\
      --results-dir results \\
      --out-dir plots \\
      --season-24 backtest_tft_long_2023_24.json \\
      --season-25 backtest_tft_long_2024_25.json

All outputs go to --out-dir (created if it does not exist).
PDF versions are suitable for LaTeX \\includegraphics; PNG for PPTX embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

# Model color map
C: dict[str, str] = {
    "rolling_avg": "#888888",
    "ewma": "#aaaaaa",
    "hmm_scalar": "#d62728",
    "kf_scalar": "#ff7f0e",
    "fused_scalar": "#e377c2",
    "mv_hmm": "#9467bd",
    "enriched": "#2ca02c",
    "enriched_mvhmm_blend": "#1f77b4",
    "tft_q50": "#17becf",
}
LABEL: dict[str, str] = {
    "rolling_avg": "Rolling Avg",
    "ewma": "EWMA",
    "hmm_scalar": "HMM (scalar)",
    "kf_scalar": "KF (scalar)",
    "fused_scalar": "Fused scalar",
    "mv_hmm": "MV-HMM",
    "enriched": "Enriched",
    "enriched_mvhmm_blend": "Enriched+MV-HMM",
    "tft_q50": "TFT q50",
}
INFERENCE_ORDER = [
    "rolling_avg",
    "ewma",
    "hmm_scalar",
    "kf_scalar",
    "fused_scalar",
    "mv_hmm",
    "enriched",
    "enriched_mvhmm_blend",
    "tft_q50",
]

OPT_STRATEGIES: dict[str, tuple[str, str]] = {
    "greedy_rolling": ("Greedy (rolling)", "#888888"),
    "ilp_ewma": ("ILP + EWMA", "#ff7f0e"),
    "ilp_fused_scalar": ("ILP + Fused", "#e377c2"),
    "ilp_mvhmm_0.0": ("ILP + MV-HMM", "#9467bd"),
    "ilp_blend_0.0": ("ILP + Blend", "#2ca02c"),
    "ilp_enriched_0.0": ("ILP + Enriched", "#1f77b4"),
    "ilp_tft_0.0": ("ILP + TFT", "#17becf"),
    "lagr_enriched": ("Lagr + Enriched", "#bcbd22"),
    "ilp_semivar_0.5": ("ILP + Semivar(λ=0.5)", "#d62728"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.{{pdf,png}}")


def ablation_barh(
    ax: plt.Axes,
    ablation: dict,
    metric: str,
    season: str,
    order: list[str],
    best_key: str,
    baseline_key: str = "rolling_avg",
) -> None:
    """Horizontal bar chart for a single ablation metric + season."""
    vals = [ablation[k][metric] for k in order if k in ablation]
    labels = [LABEL[k] for k in order if k in ablation]
    colors = [C[k] for k in order if k in ablation]
    keys = [k for k in order if k in ablation]  # noqa : F841

    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", height=0.7)

    baseline = ablation[baseline_key][metric]
    best = ablation[best_key][metric]
    pct = 100 * (baseline - best) / baseline
    ax.axvline(baseline, color="#888888", lw=1.2, ls="--", label=f"Baseline={baseline:.3f}")
    ax.axvline(best, color=C[best_key], lw=1.2, ls="--", label=f"Best={best:.3f} (−{pct:.1f}%)")

    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(f"{metric.upper()} (lower = better)", fontsize=10)
    ax.set_title(f"Inference Ablation ({metric.upper()}) — {season}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=7.5)


# ── Figure 1: Inference MSE ablation ─────────────────────────────────────────


def fig_inference_ablation(d24: dict, d25: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, (season, data) in zip(axes, [("2023-24", d24), ("2024-25", d25)]):
        ablation_barh(
            ax, data["inference"]["ablation"], "mse", season, INFERENCE_ORDER, best_key="enriched_mvhmm_blend"
        )
    plt.tight_layout()
    save(fig, out, "fig_inference_ablation")


# ── Figure 2: Inference MAE ablation ─────────────────────────────────────────


def fig_inference_mae(d24: dict, d25: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, (season, data) in zip(axes, [("2023-24", d24), ("2024-25", d25)]):
        ablation_barh(ax, data["inference"]["ablation"], "mae", season, INFERENCE_ORDER, best_key="tft_q50")
    plt.tight_layout()
    save(fig, out, "fig_inference_mae")


# ── Figure 3: Calibration ─────────────────────────────────────────────────────


def fig_calibration(d24: dict, d25: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (season, data) in zip(axes, [("2023-24", d24), ("2024-25", d25)]):
        inf = data["inference"]
        achieved = [inf["calibration_50"] * 100, inf["calibration_95"] * 100]
        ideal = [50.0, 95.0]
        x, w = np.arange(2), 0.35
        ax.bar(x - w / 2, ideal, w, label="Ideal", color="#dddddd", edgecolor="gray")
        ax.bar(x + w / 2, achieved, w, label="Achieved", color="#1f77b4", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["50% CI", "95% CI"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Coverage (%)")
        ax.set_title(f"Calibration — {season}", fontweight="bold")
        ax.legend(fontsize=9)
        for xi, a in zip(x + w / 2, achieved):
            ax.text(xi, a + 1.5, f"{a:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    save(fig, out, "fig_calibration")


# ── Figure 4: Optimization strategy comparison ────────────────────────────────


def fig_optimization_strategies(d24: dict, d25: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (season, data) in zip(axes, [("2023-24", d24), ("2024-25", d25)]):
        strats = data["optimization"]["strategies"]
        oracle = data["optimization"]["oracle_mean_per_gw"]
        keys = [k for k in OPT_STRATEGIES if k in strats]
        means = [strats[k]["mean_per_gw"] for k in keys]
        labels = [OPT_STRATEGIES[k][0] for k in keys]
        colors = [OPT_STRATEGIES[k][1] for k in keys]
        y = np.arange(len(keys))

        bars = ax.barh(y, means, color=colors, edgecolor="white", height=0.7)
        ax.axvline(oracle, color="black", lw=1.5, ls="--", label=f"Oracle={oracle:.1f}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel("Mean points / GW", fontsize=10)
        ax.set_title(f"Optimization Strategies — {season}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        for bar, v in zip(bars, means):
            pct = 100 * v / oracle
            ax.text(
                v + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} ({pct:.0f}%)",
                va="center",
                fontsize=7.5,
            )
    plt.tight_layout()
    save(fig, out, "fig_optimization_strategies")


# ── Figure 5: Risk-return trade-off ──────────────────────────────────────────


def fig_risk_tradeoff(d24: dict, d25: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = {
        "Enriched λ": {
            "keys": ["ilp_enriched_0.0", "ilp_enriched_0.5", "ilp_enriched_1.0"],
            "lambdas": [0.0, 0.5, 1.0],
            "color": "#1f77b4",
            "marker": "o",
        },
        "Semivar λ": {
            "keys": ["ilp_semivar_0.5", "ilp_semivar_1.0", "ilp_semivar_1.5"],
            "lambdas": [0.5, 1.0, 1.5],
            "color": "#d62728",
            "marker": "s",
        },
        "MV-HMM λ": {
            "keys": ["ilp_mvhmm_0.0", "ilp_mvhmm_0.5", "ilp_mvhmm_1.0"],
            "lambdas": [0.0, 0.5, 1.0],
            "color": "#9467bd",
            "marker": "^",
        },
    }
    for ax, (season, data) in zip(axes, [("2023-24", d24), ("2024-25", d25)]):
        strats = data["optimization"]["strategies"]
        for gname, gdata in groups.items():
            means_, cvs = [], []
            for k, lam in zip(gdata["keys"], gdata["lambdas"]):
                if k in strats:
                    means_.append(strats[k]["mean_per_gw"])
                    cvs.append(strats[k]["cv"])
            if means_:
                ax.plot(cvs, means_, marker=gdata["marker"], color=gdata["color"], label=gname, lw=1.5, ms=7)
                for lam, cv_, m in zip(gdata["lambdas"], cvs, means_):
                    ax.annotate(f"λ={lam}", (cv_, m), textcoords="offset points", xytext=(4, 2), fontsize=7.5)
        ax.set_xlabel("Coefficient of Variation (lower = more stable)", fontsize=10)
        ax.set_ylabel("Mean points / GW", fontsize=10)
        ax.set_title(f"Risk-Return Trade-off — {season}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out, "fig_risk_tradeoff")


# ── Figure 6: Lagrangian vs ILP ───────────────────────────────────────────────


def fig_lagrangian_vs_ilp(d24: dict, d25: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    seasons = ["2023-24", "2024-25"]
    ilp_pts = [d["optimization"]["strategies"]["ilp_enriched_0.0"]["total_points"] for d in (d24, d25)]
    lagr_pts = [d["optimization"]["strategies"]["lagr_enriched"]["total_points"] for d in (d24, d25)]
    oracle_pts = [d["optimization"]["oracle_total"] for d in (d24, d25)]

    x, w = np.arange(2), 0.25
    ax.bar(x - w, ilp_pts, w, label="ILP + Enriched", color="#1f77b4")
    ax.bar(x, lagr_pts, w, label="Lagr. + Enriched", color="#bcbd22")
    ax.bar(x + w, oracle_pts, w, label="Oracle (hindsight)", color="#2ca02c", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Total Points (33 GWs)")
    ax.set_title("ILP vs Lagrangian Relaxation vs Oracle", fontweight="bold")
    ax.legend()

    for xi, vals in zip(x, zip(ilp_pts, lagr_pts, oracle_pts)):
        for di, (v, bar_x) in enumerate(zip(vals, [xi - w, xi, xi + w])):
            ax.text(bar_x, v + 20, f"{int(v)}", ha="center", fontsize=8.5)

    for i in range(2):
        pct = 100 * lagr_pts[i] / ilp_pts[i]
        ax.annotate(
            f"{pct:.1f}% of ILP",
            xy=(i, lagr_pts[i]),
            xytext=(i + 0.22, lagr_pts[i] - 180),
            fontsize=8,
            color="#7a6800",
            arrowprops=dict(arrowstyle="->", color="#7a6800", lw=0.8),  # noqa : C408 noqa: E501
        )
    plt.tight_layout()
    save(fig, out, "fig_lagrangian_vs_ilp")


# ── Figure 7: Summary table ───────────────────────────────────────────────────


def fig_summary_table(d24: dict, d25: dict, out: Path) -> None:
    s24 = d24["optimization"]["strategies"]
    s25 = d25["optimization"]["strategies"]
    o24 = d24["optimization"]["oracle_total"]
    o25 = d25["optimization"]["oracle_total"]

    rows_data = [
        ("Greedy (rolling avg)", s24["greedy_rolling"], s25["greedy_rolling"]),
        ("ILP + EWMA", s24["ilp_ewma"], s25["ilp_ewma"]),
        ("ILP + MV-HMM", s24["ilp_mvhmm_0.0"], s25["ilp_mvhmm_0.0"]),
        ("ILP + Enriched (λ=0)", s24["ilp_enriched_0.0"], s25["ilp_enriched_0.0"]),
        ("ILP + Semivar (λ=0.5)", s24["ilp_semivar_0.5"], s25["ilp_semivar_0.5"]),
        ("ILP + Blend (λ=0)", s24["ilp_blend_0.0"], s25["ilp_blend_0.0"]),
        ("Lagr + Enriched", s24["lagr_enriched"], s25["lagr_enriched"]),
    ]
    header = ["Strategy", "2023-24 pts", "% Oracle", "2024-25 pts", "% Oracle"]
    cell_text = []
    for name, r24, r25 in rows_data:
        cell_text.append([
            name,
            str(int(r24["total_points"])),
            f"{r24['pct_of_oracle']:.1f}%",
            str(int(r25["total_points"])),
            f"{r25['pct_of_oracle']:.1f}%",
        ])
    cell_text.append(["Oracle (hindsight)", str(int(o24)), "100%", str(int(o25)), "100%"])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    # Highlight enriched row (index 3, 0-based after header)
    for j in range(5):
        tbl[4, j].set_facecolor("#d4eaff")
        tbl[4, j].set_text_props(fontweight="bold")
    # Oracle row
    for j in range(5):
        tbl[8, j].set_facecolor("#e8ffe8")
    ax.set_title(
        "Optimization Strategy Summary (GW6–38, 33 gameweeks)", fontweight="bold", fontsize=11, pad=12
    )
    plt.tight_layout()
    save(fig, out, "fig_summary_table")


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate FPLX backtest plots")
    p.add_argument("--results-dir", default="results", help="Directory containing the JSON result files")
    p.add_argument("--season-24", default="backtest_tft_long_2023_24.json")
    p.add_argument("--season-25", default="backtest_tft_long_2024_25.json")
    p.add_argument("--out-dir", default="plots", help="Output directory for PDF and PNG figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r_dir = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(r_dir / args.season_24) as f:
        d24 = json.load(f)
    with open(r_dir / args.season_25) as f:
        d25 = json.load(f)

    print(f"Generating plots → {out}/")
    fig_inference_ablation(d24, d25, out)
    fig_inference_mae(d24, d25, out)
    fig_calibration(d24, d25, out)
    fig_optimization_strategies(d24, d25, out)
    fig_risk_tradeoff(d24, d25, out)
    fig_lagrangian_vs_ilp(d24, d25, out)
    fig_summary_table(d24, d25, out)
    print("Done.")


if __name__ == "__main__":
    main()
