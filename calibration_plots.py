"""
Calibration checks for a generative ensemble model vs ERA5 ground truth.

Ground truth : data/ERA5_all_wind_test.pt
Ensemble     : data/results/ERA5_ERA5_{m}.pt  for m in 0..9

Expected tensor shapes (adapt if needed):
  ground_truth : (N, C, H, W)   – N samples, C channels, spatial H×W
  each member  : (N, C, H, W)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GT_PATH       = Path("data/ERA5_all_wind_test.pt")
MEMBERS_GLOB  = "data/results/ERA5_ERA5_{m}_11.pt"
N_MEMBERS     = 10
OUTPUT_DIR    = Path("data/outputs_11")
CHANNEL_NAMES = ['sfcWind', 'uas', 'vas', 'sfcWindmax']      # e.g. ["U10", "V10"] – set to None to auto-label
# ──────────────────────────────────────────────────────────────────────────────

# ── Timeseries spread plot config ─────────────────────────────────────────────
PLOT_H        = 22            # spatial index along H axis
PLOT_W        = 32            # spatial index along W axis
PLOT_T_START  = 1000             # first time index (sample)
PLOT_T_END    = 1030            # last time index  (sample, exclusive)
# ──────────────────────────────────────────────────────────────────────────────
 

OUTPUT_DIR.mkdir(exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    print("Loading ground truth …")
    gt = torch.load(GT_PATH, map_location="cpu")
    if not isinstance(gt, torch.Tensor):
        raise TypeError(f"Expected Tensor for ground truth, got {type(gt)}")

    members = []
    for m in range(N_MEMBERS):
        path = Path(MEMBERS_GLOB.format(m=m))
        print(f"  Loading member {m} from {path}")
        t = torch.load(path, map_location="cpu")
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor for member {m}, got {type(t)}")
        members.append(t)

    # Stack → (M, N, C, H, W)
    ensemble = torch.stack(members, dim=0).float()
    gt       = gt.float()

    # ── Denormalise ensemble using gt statistics (per channel) ────────────────
    # mean/std computed over (N, H, W) for each channel c → shape (C,)
    gt_mean = gt.mean(dim=(0, 2, 3))   # (C,)
    gt_std  = gt.std(dim=(0, 2, 3))    # (C,)
    print("\nDenormalising ensemble with gt stats:")
    for c, (m, s) in enumerate(zip(gt_mean.tolist(), gt_std.tolist())):
        print(f"  channel {c}: mean={m:.4f}  std={s:.4f}")

    # broadcast: (M, N, C, H, W) * (1, 1, C, 1, 1)
    shape = (1, 1, -1, 1, 1)
    ensemble = ensemble * gt_std.view(shape) + gt_mean.view(shape)
    # gt is already in physical units — no transform needed

    print(f"\nGround truth shape : {gt.shape}")
    print(f"Ensemble shape     : {ensemble.shape}  (members, samples, …)")
    return gt, ensemble


# ── 2. CRPS ───────────────────────────────────────────────────────────────────
def crps_ensemble(obs: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """
    Compute the ensemble CRPS.
    Uses the energy-score decomposition:
        CRPS = E|X - y| - 0.5 * E|X - X'|

    Parameters
    ----------
    obs      : (...,)        ground-truth observations
    ensemble : (M, ...)      M ensemble members, same trailing shape as obs

    Returns
    -------
    crps     : (...)         per-point CRPS
    """
    M = ensemble.shape[0]

    # E|X - y|
    term1 = np.mean(np.abs(ensemble - obs[np.newaxis]), axis=0)

    # E|X - X'|  (unbiased estimator)
    # sum_{i<j} |x_i - x_j|  = 0.5 * (M * sum|x_i| - (sum x_i)^2) … or brute-force
    diff_sum = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            diff_sum += np.abs(ensemble[i] - ensemble[j])
    term2 = diff_sum / (M * (M - 1) / 2) * 0.5

    return term1 - term2


def compute_crps(gt: torch.Tensor, ensemble: torch.Tensor):
    """
    Returns per-channel mean CRPS and the full spatial map.
    gt       : (N, C, H, W)
    ensemble : (M, N, C, H, W)
    """
    gt_np  = gt.numpy()            # (N, C, H, W)
    ens_np = ensemble.numpy()      # (M, N, C, H, W)

    C = gt_np.shape[1]
    crps_maps   = []   # (C, H, W)
    crps_scalar = []   # (C,)

    for c in range(C):
        obs_c = gt_np[:, c, :, :]      # (N, H, W)
        ens_c = ens_np[:, :, c, :, :]  # (M, N, H, W)
        crps_c = crps_ensemble(obs_c, ens_c)       # (N, H, W)
        crps_maps.append(crps_c.mean(axis=0))      # (H, W)
        crps_scalar.append(float(crps_c.mean()))

    return np.array(crps_maps), np.array(crps_scalar)


# ── 3. Rank / PIT histogram ───────────────────────────────────────────────────
def rank_histogram(gt: torch.Tensor, ensemble: torch.Tensor):
    """
    Talagrand diagram (rank histogram).
    Returns rank counts, shape (M+1, C).
    """
    gt_np  = gt.numpy()   # (N, C, H, W)
    ens_np = ensemble.numpy()  # (M, N, C, H, W)
    M, N, C, H, W = ens_np.shape

    ranks_per_channel = []
    for c in range(C):
        obs_flat = gt_np[:, c].reshape(-1)          # (N*H*W,)
        ens_flat = ens_np[:, :, c].reshape(M, -1)   # (M, N*H*W)

        # rank of obs among (ens_flat sorted + obs)
        # = number of members strictly below obs
        below = (ens_flat < obs_flat[np.newaxis]).sum(axis=0)  # (N*H*W,)
        counts = np.bincount(below, minlength=M + 1)
        ranks_per_channel.append(counts)

    return np.array(ranks_per_channel)   # (C, M+1)


# ── 4. Spread–Skill ───────────────────────────────────────────────────────────
def spread_skill(gt: torch.Tensor, ensemble: torch.Tensor):
    """
    Returns (spread, rmse) per channel, aggregated over all spatial points.
    spread = std of ensemble members
    skill  = RMSE of ensemble mean vs obs
    """
    gt_np  = gt.numpy()
    ens_np = ensemble.numpy()
    C = gt_np.shape[1]

    spreads, rmses = [], []
    for c in range(C):
        obs  = gt_np[:, c]                    # (N, H, W)
        ens  = ens_np[:, :, c]               # (M, N, H, W)
        mean = ens.mean(axis=0)              # (N, H, W)
        std  = ens.std(axis=0)              # (N, H, W)

        rmse   = float(np.sqrt(((mean - obs) ** 2).mean()))
        spread = float(std.mean())
        spreads.append(spread)
        rmses.append(rmse)

    return np.array(spreads), np.array(rmses)


# ── 5. Reliability diagram (marginal coverage) ────────────────────────────────
def reliability_diagram(gt: torch.Tensor, ensemble: torch.Tensor,
                        alphas=None):
    """
    For each nominal coverage level α, compute the empirical coverage of the
    central [α] prediction interval.
    Returns (nominal, empirical) per channel.
    """
    if alphas is None:
        alphas = np.linspace(0.05, 0.95, 19)

    gt_np  = gt.numpy()
    ens_np = ensemble.numpy()
    M, N, C, H, W = ens_np.shape

    results = []  # per channel: (alphas, empirical_coverages)
    for c in range(C):
        obs  = gt_np[:, c].reshape(-1)         # (N*H*W,)
        ens  = ens_np[:, :, c].reshape(M, -1)  # (M, N*H*W)

        empirical = []
        for a in alphas:
            lo = np.quantile(ens, (1 - a) / 2, axis=0)
            hi = np.quantile(ens, (1 + a) / 2, axis=0)
            coverage = float(((obs >= lo) & (obs <= hi)).mean())
            empirical.append(coverage)

        results.append(np.array(empirical))

    return alphas, results


# ── 6. Plotting ───────────────────────────────────────────────────────────────
def make_plots(gt, ensemble, crps_maps, crps_scalar,
               rank_counts, spreads, rmses,
               alphas, reliability):

    C = gt.shape[1]
    ch_names = CHANNEL_NAMES or [f"Ch {c}" for c in range(C)]

    # ── Figure 1: CRPS spatial maps ──────────────────────────────────────────
    fig, axes = plt.subplots(1, C, figsize=(5 * C, 4))
    if C == 1:
        axes = [axes]
    fig.suptitle("Mean CRPS — spatial map", fontsize=14, fontweight="bold")
    for c, ax in enumerate(axes):
        im = ax.imshow(crps_maps[c], origin="upper", cmap="plasma")
        ax.set_title(f"{ch_names[c]}  |  mean CRPS = {crps_scalar[c]:.4f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "crps_spatial.png", dpi=150)
    plt.close()
    print("Saved: outputs/crps_spatial.png")

    # ── Figure 2: Rank histogram ──────────────────────────────────────────────
    M = ensemble.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(5 * C, 3.5))
    if C == 1:
        axes = [axes]
    fig.suptitle("Rank (Talagrand) histogram", fontsize=14, fontweight="bold")
    uniform = np.ones(M + 1) / (M + 1)
    for c, ax in enumerate(axes):
        counts = rank_counts[c] / rank_counts[c].sum()
        ax.bar(range(M + 1), counts, color="#4C72B0", alpha=0.75, label="Observed")
        ax.axhline(uniform[0], color="red", linestyle="--", lw=1.5, label="Uniform")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Relative frequency")
        ax.set_title(ch_names[c])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rank_histogram.png", dpi=150)
    plt.close()
    print("Saved: outputs/rank_histogram.png")

    # ── Figure 3: Spread-Skill ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(C)
    w = 0.35
    ax.bar(x - w / 2, spreads, w, label="Ensemble spread (std)", color="#55A868")
    ax.bar(x + w / 2, rmses,   w, label="RMSE of ensemble mean",  color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names)
    ax.set_ylabel("Value")
    ax.set_title("Spread – Skill diagram", fontweight="bold")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spread_skill.png", dpi=150)
    plt.close()
    print("Saved: outputs/spread_skill.png")

    # ── Figure 4: Reliability diagram ────────────────────────────────────────
    fig, axes = plt.subplots(1, C, figsize=(5 * C, 4))
    if C == 1:
        axes = [axes]
    fig.suptitle("Reliability diagram (marginal coverage)", fontsize=14, fontweight="bold")
    for c, ax in enumerate(axes):
        ax.plot(alphas, reliability[c], "o-", color="#4C72B0", label="Empirical coverage")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.05, color="black")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(ch_names[c])
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reliability.png", dpi=150)
    plt.close()
    print("Saved: outputs/reliability.png")

    # ── Figure 5: Summary dashboard ──────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Calibration Dashboard — ERA5 ensemble", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # CRPS bar
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(ch_names, crps_scalar, color="#8172B2")
    ax0.set_title("Mean CRPS per channel")
    ax0.set_ylabel("CRPS")

    # Spread-skill
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x - w/2, spreads, w, label="Spread", color="#55A868")
    ax1.bar(x + w/2, rmses,   w, label="RMSE",   color="#C44E52")
    ax1.set_xticks(x); ax1.set_xticklabels(ch_names)
    ax1.set_title("Spread – Skill"); ax1.legend(fontsize=8)

    # Rank histogram (first channel)
    ax2 = fig.add_subplot(gs[1, 0])
    counts0 = rank_counts[0] / rank_counts[0].sum()
    ax2.bar(range(M + 1), counts0, color="#4C72B0", alpha=0.75)
    ax2.axhline(1 / (M + 1), color="red", linestyle="--", lw=1.5)
    ax2.set_title(f"Rank histogram — {ch_names[0]}")
    ax2.set_xlabel("Rank"); ax2.set_ylabel("Rel. freq.")

    # Reliability (first channel)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(alphas, reliability[0], "o-", color="#4C72B0")
    ax3.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_title(f"Reliability — {ch_names[0]}")
    ax3.set_xlabel("Nominal"); ax3.set_ylabel("Empirical")
    ax3.set_aspect("equal")

    plt.savefig(OUTPUT_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/dashboard.png")


# ── 7. Summary table ─────────────────────────────────────────────────────────

# ── 7. Timeseries spread plot ─────────────────────────────────────────────
def plot_timeseries_spread(gt, ensemble,
                           h=None, w=None,
                           t_start=None, t_end=None):
    """
    For a single spatial point (h, w), plot ground truth vs ensemble spread
    over a time window [t_start, t_end).

    gt       : (N, C, H, W)
    ensemble : (M, N, C, H, W)
    """
    h       = PLOT_H       if h       is None else h
    w       = PLOT_W       if w       is None else w
    t_start = PLOT_T_START if t_start is None else t_start
    t_end   = PLOT_T_END   if t_end   is None else t_end

    gt_np  = gt.numpy()        # (N, C, H, W)
    ens_np = ensemble.numpy()  # (M, N, C, H, W)

    t_end    = min(t_end, gt_np.shape[0])
    times    = np.arange(t_start, t_end)
    C        = gt_np.shape[1]
    ch_names = CHANNEL_NAMES or [f"Ch {c}" for c in range(C)]

    fig, axes = plt.subplots(C, 1, figsize=(12, 3.5 * C), sharex=True)
    if C == 1:
        axes = [axes]

    fig.suptitle(
        f"Ensemble spread vs ground truth  —  point (h={h}, w={w}), "
        f"t=[{t_start}, {t_end})",
        fontsize=13, fontweight="bold"
    )

    quantile_bands = [(0.05, 0.95, 0.18, "5–95 %"),
                      (0.25, 0.75, 0.32, "25–75 %")]

    for c, ax in enumerate(axes):
        obs = gt_np[t_start:t_end, c, h, w]        # (T,)
        ens = ens_np[:, t_start:t_end, c, h, w]    # (M, T)

        # individual members (thin, semi-transparent)
        for m in range(ens.shape[0]):
            ax.plot(times, ens[m], color="#4C72B0", alpha=0.15, lw=0.8)

        # shaded quantile bands (wide then narrow)
        for lo_q, hi_q, a, lbl in quantile_bands:
            lo = np.quantile(ens, lo_q, axis=0)
            hi = np.quantile(ens, hi_q, axis=0)
            ax.fill_between(times, lo, hi, alpha=a, color="#4C72B0", label=lbl)

        # ensemble median
        median = np.median(ens, axis=0)
        ax.plot(times, median, color="#4C72B0", lw=1.8, label="Ensemble median")

        # ground truth
        ax.plot(times, obs, color="#C44E52", lw=2, zorder=5, label="Ground truth")

        ax.set_ylabel(ch_names[c])
        ax.legend(fontsize=8, loc="upper right", ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time index")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "timeseries_spread.png", dpi=150)
    plt.close()
    print("Saved: outputs/timeseries_spread.png")

def print_summary(crps_scalar, spreads, rmses, rank_counts):
    C = len(crps_scalar)
    ch_names = CHANNEL_NAMES or [f"Ch {c}" for c in range(C)]
    M = rank_counts.shape[1] - 1

    print("\n" + "=" * 60)
    print(f"{'Channel':<10} {'CRPS':>10} {'Spread':>10} {'RMSE':>10} {'S/S ratio':>10}")
    print("-" * 60)
    for c in range(C):
        ratio = spreads[c] / rmses[c] if rmses[c] > 0 else float("nan")
        print(f"{ch_names[c]:<10} {crps_scalar[c]:>10.4f} {spreads[c]:>10.4f} "
              f"{rmses[c]:>10.4f} {ratio:>10.3f}")

    print("=" * 60)
    print("\nSpread/Skill ratio: 1.0 = perfectly calibrated spread")
    print("Rank histogram flat = well-calibrated ensemble")
    print()

    # Basic rank histogram diagnosis
    for c in range(C):
        counts = rank_counts[c] / rank_counts[c].sum()
        u_shaped = counts[0] + counts[-1] > counts[M // 2] * 1.5
        dome     = counts[M // 2] > counts[0] * 1.5
        if u_shaped:
            print(f"  [{ch_names[c]}] U-shaped rank hist → ensemble likely UNDER-dispersed")
        elif dome:
            print(f"  [{ch_names[c]}] dome-shaped rank hist → ensemble likely OVER-dispersed")
        else:
            print(f"  [{ch_names[c]}] rank histogram looks roughly flat ✓")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gt, ensemble = load_data()

    print("\n[1/4] Computing CRPS …")
    crps_maps, crps_scalar = compute_crps(gt, ensemble)

    print("[2/4] Computing rank histograms …")
    rank_counts = rank_histogram(gt, ensemble)

    print("[3/4] Computing spread–skill …")
    spreads, rmses = spread_skill(gt, ensemble)

    print("[4/4] Computing reliability diagrams …")
    alphas, reliability = reliability_diagram(gt, ensemble)

    print_summary(crps_scalar, spreads, rmses, rank_counts)

    print("\nGenerating plots …")
    make_plots(gt, ensemble, crps_maps, crps_scalar,
               rank_counts, spreads, rmses,
               alphas, reliability)

    print("Generating timeseries spread plot …")
    plot_timeseries_spread(gt, ensemble)

    print("\nDone! All outputs saved to outputs/")