"""
make_paper_figures.py
=====================

Regenerate the four publication figures used in the IEEE conference paper
and the comprehensive report, directly from `bench_results.pkl`.

Outputs (written to the same directory as this script):
    fig_scaling.png       — training time vs N, log-log, both BVPs
    fig_speedup.png       — speedup over M0 vs N, log-log, both BVPs
    fig_solution.png      — solution overlay and pointwise error at N=1000
    fig_convergence.png   — LM residual trace per outer iteration

The cache must contain a 'results' list where each entry has keys
'N', 'problem' ('linear' | 'nonlinear'), and per-method dicts 'M0', 'M1', 'M2'
each carrying 'train_time' (seconds), 'l2_error', and (for M1/M2) 'losses'.

Run:
    python make_paper_figures.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- styling ---------------------------------

# Color and marker conventions used consistently across all figures
STYLE = {
    "M0": {"color": "#1f77b4", "marker": "o", "label": "M0 (Naive)"},
    "M1": {"color": "#ff7f0e", "marker": "s", "label": "M1 (SMW)"},
    "M2": {"color": "#2ca02c", "marker": "^", "label": "M2 (LSQR)"},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ----------------------------- data loading ----------------------------


def load_cache(cache_path: Path) -> dict:
    """Load and validate the benchmark cache."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if "results" not in cache:
        raise KeyError(f"'results' key missing from {cache_path}")
    return cache


def extract_curves(results: list[dict], problem: str) -> dict:
    """Return {method: (Ns, times, errors)} sorted by N for one BVP."""
    out: dict = {m: {"N": [], "t": [], "err": []} for m in ("M0", "M1", "M2")}
    for e in results:
        if e["problem"] != problem:
            continue
        for m in ("M0", "M1", "M2"):
            out[m]["N"].append(e["N"])
            out[m]["t"].append(e[m]["train_time"] * 1e3)  # → milliseconds
            out[m]["err"].append(e[m]["l2_error"])
    for m in out:
        order = np.argsort(out[m]["N"])
        for k in ("N", "t", "err"):
            out[m][k] = np.array(out[m][k])[order]
    return out


# ----------------------------- figures ---------------------------------


def fig_scaling(results: list[dict], outpath: Path) -> None:
    """
    2x2 grid: rows = (relative L^2 error, training time);
              cols = (linear BVP, nonlinear BVP).
    Time panels overlay O(N) and O(N^3) reference slopes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    for col, problem in enumerate(("linear", "nonlinear")):
        d = extract_curves(results, problem)

        # ----- top row: relative L^2 error -----
        ax = axes[0, col]
        for m in ("M0", "M1", "M2"):
            s = STYLE[m]
            ax.loglog(d[m]["N"], np.clip(d[m]["err"], 1e-16, None),
                      color=s["color"], marker=s["marker"], label=s["label"])
        # mark the LAPACK gelsd failure at N=2000 on the linear BVP
        if problem == "linear":
            bad = (d["M0"]["N"] == 2000) & (d["M0"]["err"] > 1e-2)
            if bad.any():
                ax.plot(d["M0"]["N"][bad], d["M0"]["err"][bad],
                        marker="x", color="C0", markersize=12,
                        markeredgewidth=2.5, linestyle="None", zorder=5)
        ax.set_xlabel("$N$ (CGL nodes)")
        ax.set_ylabel(r"$\|u_h - u^{\star}\|_2 / \|u^{\star}\|_2$")
        title = "Linear BVP" if problem == "linear" else "Nonlinear BVP"
        ax.set_title(f"{title} — relative $L^2$ error")
        ax.legend(loc="best")

        # ----- bottom row: training time -----
        ax = axes[1, col]
        for m in ("M0", "M1", "M2"):
            s = STYLE[m]
            ax.loglog(d[m]["N"], d[m]["t"],
                      color=s["color"], marker=s["marker"], label=s["label"])

        # Reference O(N) and O(N^3) slopes anchored to M0 at N=250
        anchor_N, anchor_t = 250, d["M0"]["t"][d["M0"]["N"] == 250]
        if anchor_t.size:
            anchor_t = float(anchor_t[0])
            Ns_ref = np.array([d["M0"]["N"].min(), d["M0"]["N"].max()],
                              dtype=float)
            ax.loglog(Ns_ref, anchor_t * (Ns_ref / anchor_N) ** 3,
                      "k:", lw=0.9, alpha=0.7, label=r"$\mathcal{O}(N^3)$")
            ax.loglog(Ns_ref, anchor_t * (Ns_ref / anchor_N),
                      "k--", lw=0.9, alpha=0.7, label=r"$\mathcal{O}(N)$")
        ax.set_xlabel("$N$ (CGL nodes)")
        ax.set_ylabel("time (ms)")
        ax.set_title(f"{title} — training time")
        ax.legend(loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def fig_speedup(results: list[dict], outpath: Path) -> None:
    """Speedup factor t_M0 / t_method versus N for M1 and M2, both BVPs."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for col, problem in enumerate(("linear", "nonlinear")):
        d = extract_curves(results, problem)
        ax = axes[col]
        for m in ("M1", "M2"):
            speedup = d["M0"]["t"] / d[m]["t"]
            s = STYLE[m]
            ax.loglog(d[m]["N"], speedup,
                      color=s["color"], marker=s["marker"],
                      label=f"{s['label']} speedup")

        ax.axhline(1.0, color="black", ls=":", lw=0.9, alpha=0.6)
        title = "Linear BVP" if problem == "linear" else "Nonlinear BVP"
        ax.set_xlabel("$N$ (CGL nodes)")
        ax.set_ylabel(r"$t_{\mathrm{M0}} / t_{\mathrm{method}}$")
        ax.set_title(f"{title} — speedup over naive baseline")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def fig_solution(results: list[dict], outpath: Path,
                 target_N: int = 1000) -> None:
    """
    Two-row plot for both BVPs:
      top:    M1 solution overlaid on the manufactured truth
      bottom: pointwise absolute error in log scale
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))

    for col, problem in enumerate(("linear", "nonlinear")):
        entry = next((e for e in results
                      if e["N"] == target_N and e["problem"] == problem), None)
        if entry is None:
            print(f"  ** skipping {problem}: no entry at N={target_N}")
            continue

        x       = np.asarray(entry["M1"]["x"])
        u_pred  = np.asarray(entry["M1"]["u_pred"])
        u_true  = np.asarray(entry["M1"]["u_true"])
        abs_err = np.abs(u_pred - u_true)

        ax = axes[0, col]
        ax.plot(x, u_true, "k-", lw=2.5, label="$u^{\\star}(x) = \\sin(\\pi x)$",
                alpha=0.5)
        ax.plot(x, u_pred, color=STYLE["M1"]["color"], lw=1.4,
                linestyle="--", label="M1 (SMW)")
        title = "Linear BVP" if problem == "linear" else "Nonlinear BVP"
        ax.set_title(f"{title}  ($N={target_N}$)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(x)$")
        ax.legend(loc="lower center")

        ax = axes[1, col]
        ax.semilogy(x, np.clip(abs_err, 1e-16, None),
                    color=STYLE["M1"]["color"], lw=1.0)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$|u_h(x) - u^{\star}(x)|$")
        ax.set_title("Pointwise absolute error (log scale)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


def fig_convergence(results: list[dict], outpath: Path,
                    target_Ns: tuple = (40, 250, 1000)) -> None:
    """LM residual trace versus accepted outer iteration for M1 and M2."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    cmap = plt.cm.viridis
    for col, problem in enumerate(("linear", "nonlinear")):
        ax = axes[col]
        for j, N in enumerate(target_Ns):
            entry = next((e for e in results
                          if e["N"] == N and e["problem"] == problem), None)
            if entry is None:
                continue
            color = cmap(0.15 + 0.7 * j / max(1, len(target_Ns) - 1))
            for m, ls in (("M1", "-"), ("M2", "--")):
                losses = entry[m].get("losses")
                if losses is None or len(losses) < 2:
                    continue
                ax.semilogy(np.arange(len(losses)), losses,
                            ls, color=color, marker=STYLE[m]["marker"],
                            markersize=4, alpha=0.85,
                            label=f"{m}, $N={N}$")
        title = "Linear BVP" if problem == "linear" else "Nonlinear BVP"
        ax.set_title(f"{title} — LM convergence")
        ax.set_xlabel("accepted outer iteration")
        ax.set_ylabel(r"$\|F(u_{\theta})\|_2$")
        ax.legend(loc="best", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}")


# ----------------------------- main ------------------------------------


def main() -> None:
    here = Path(__file__).resolve().parent
    cache_path = here / "bench_results.pkl"
    cache = load_cache(cache_path)
    results = cache["results"]
    print(f"Loaded {len(results)} entries from {cache_path.name}")

    fig_scaling   (results, here / "fig_scaling.png")
    fig_speedup   (results, here / "fig_speedup.png")
    fig_solution  (results, here / "fig_solution.png")
    fig_convergence(results, here / "fig_convergence.png")


if __name__ == "__main__":
    main()
