"""Grayscale publication figures for the results section."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import START, TRAIN_END, CAL_END, SITES

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "mathtext.fontset": "dejavuserif"})
GR = ["#000000", "#555555", "#8C8C8C", "#B5B5B5"]
MK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", ":", "-."]


def _periods(ax, days):
    t0 = pd.Timestamp(START)
    for a, b, c in [(pd.Timestamp(TRAIN_END), pd.Timestamp(CAL_END), "#E6E6E6"), (pd.Timestamp(CAL_END), days[-1], "#F3F3F3")]:
        ax.axvspan(a, b, color=c, lw=0, zorder=0)


def fig_series(P, path):
    days = P["days"]; tr = P["truth"]
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.0), dpi=300, sharex=True, gridspec_kw=dict(hspace=0.18))
    ax = axs[0]; _periods(ax, days)
    for s in range(tr.shape[0]):
        ax.plot(days, tr[s, :, 0], color="#9A9A9A", lw=0.35)
    ax.plot(days, np.nanmedian(tr[:, :, 0], 0), color="black", lw=1.0, label="Median of 10 sites")
    ax.set_ylabel("SARS-CoV-2\nlog$_{10}$ rel. median")
    lo, hi = np.nanmin(tr[:, :, 0]), np.nanmax(tr[:, :, 0])
    ax.set_ylim(lo - 0.2, hi + 1.6)
    ax.legend(loc="upper left", fontsize=6, borderaxespad=0.1)
    yt = ax.get_ylim()[1] - 0.35
    ax.text(pd.Timestamp("2023-03-31"), yt, "cal.", fontsize=5.5, ha="center", color="#444444")
    ax.text(pd.Timestamp("2023-09-15"), yt, "test", fontsize=5.5, ha="center", color="#444444")
    ax = axs[1]; _periods(ax, days)
    ax.plot(days, tr[0, :, 1], color="black", lw=0.8, label="Influenza A+B")
    ax.plot(days, tr[0, :, 2], color="#7A7A7A", lw=0.8, ls="--", label="RSV")
    ax.set_ylabel("Ottawa\nlog$_{10}$ rel. median")
    ax.set_ylim(np.nanmin(tr[0, :, 1:]) - 0.2, np.nanmax(tr[0, :, 1:]) + 1.3)
    ax.legend(loc="upper left", fontsize=6, ncol=2, borderaxespad=0.1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for a in axs:
        a.margins(x=0)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_corruption(sw, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), dpi=300, gridspec_kw=dict(wspace=0.5))
    x = sw.corruption.values * 100
    ax = axs[0]
    for k, (col, lab) in enumerate([("f1_soft", "Soft (argmax)"), ("f1_fs", "Fellegi–Sunter"), ("f1_det", "Deterministic")]):
        ax.plot(x, sw[col], color=GR[k], marker=MK[k], ms=3, lw=0.9, ls=LS[k], label=lab)
    ax.set_xlabel("Identifier corruption (%)"); ax.set_ylabel("Linkage F1"); ax.set_ylim(0.3, 1.03)
    ax.legend(fontsize=5.6, loc="lower left", borderaxespad=0.2)
    ax.set_title("(a) Linkage quality", fontsize=7)
    ax = axs[1]
    series = [("auc_cohere", "COHERE-Resp (soft)"), ("auc_cohere_hard", "Thresholded links"),
              ("auc_cohere_det", "Deterministic links"), ("auc_lgbm_soft", "LightGBM, $\\pi$-weighted"),
              ("auc_lgbm_det", "LightGBM, deterministic")]
    for k, (col, lab) in enumerate(series):
        if col in sw:
            ax.plot(x, sw[col], color=GR[k % 4], marker=MK[k], ms=3, lw=0.9, ls=LS[k % 4], label=lab)
    ax.set_xlabel("Identifier corruption (%)"); ax.set_ylabel("Test AUROC (72 h)")
    lo = np.nanmin(sw[[c for c, _ in series if c in sw]].values)
    ax.set_ylim(lo - 0.006, np.nanmax(sw[[c for c, _ in series if c in sw]].values) + 0.006)
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, handlelength=1.8, columnspacing=0.8)
    ax.set_title("(b) Downstream discrimination", fontsize=7)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_pressure(sub, null, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), dpi=300, gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.55))
    ax = axs[0]
    y = np.arange(len(sub))[::-1]
    ax.errorbar(sub.delta * 100, y, xerr=[np.clip(sub.delta - sub.ci_lo, 0, None) * 100, np.clip(sub.ci_hi - sub.delta, 0, None) * 100], fmt="o",
                color="black", ms=3, lw=0.8, capsize=1.5)
    ax.axvline(0, color="#777777", lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([f"{g} (n={n:,})" for g, n in zip(sub.group, sub.n)], fontsize=5.6)
    ax.set_xlabel("$\\Delta$AUROC $\\times 100$ (with vs without pressure)")
    ax.set_title("(a) Gain from pressure by subgroup", fontsize=7)
    ax = axs[1]
    if len(null):
        w = 0.35; xs = np.arange(len(null))
        ax.bar(xs - w / 2, null.auc_nopress, w, color="#BDBDBD", edgecolor="black", lw=0.5, label="No pressure")
        ax.bar(xs + w / 2, null.auc_press, w, color="#555555", edgecolor="black", lw=0.5, label="Gated pressure")
        lo = min(null.auc_nopress.min(), null.auc_press.min()); hi = max(null.auc_nopress.max(), null.auc_press.max())
        ax.set_ylim(lo - 0.01, hi + 0.012)
        ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in null.seed])
        ax.set_ylabel("Test AUROC")
        ax.legend(fontsize=5.4, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0)
    ax.set_title("(b) Null-pressure world", fontsize=7)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_facility(ev, P, path, sites=(0, 4), conf=None):
    te = ev["te"]; days = P["days"]
    org = np.array([r["d"] for r in te])
    fig, axs = plt.subplots(len(sites), 1, figsize=(3.45, 2.7), dpi=300, sharex=True, gridspec_kw=dict(hspace=0.25))
    for ax, s in zip(axs, sites):
        y7 = np.array([r["y"][s].sum() for r in te]); m7 = np.array([r["mint"][s].sum() for r in te])
        b7 = np.array([r["td"][s].sum() for r in te])
        dd = days[org]
        if conf is not None:
            q = conf[s] if np.ndim(conf) else conf
            ax.fill_between(dd, np.clip(m7 - q, 0, None), m7 + q, color="#DDDDDD", lw=0, label="90% interval")
        ax.plot(dd, y7, color="black", lw=0.9, label="Observed")
        ax.plot(dd, m7, color="#444444", lw=0.9, ls="--", label="COHERE-Resp (reconciled)")
        ax.plot(dd, b7, color="#8C8C8C", lw=0.8, ls=":", label="Direct ridge + wastewater")
        ax.set_ylabel("O$_2$ starts, next 7 d")
        ax.set_title(SITES[s], fontsize=6.6, loc="left", pad=1.5)
        top = max(y7.max(), m7.max(), (m7 + (q if conf is not None else 0)).max())
        ax.set_ylim(0, top * 1.42)
    axs[0].legend(fontsize=5.4, ncol=2, loc="upper left", borderaxespad=0.1, columnspacing=1.0)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 8, 10, 12])); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_horizon(tabs, path):
    """tabs: dict method -> array of per-horizon MAE (mean over cohorts)."""
    fig, ax = plt.subplots(figsize=(3.45, 1.9), dpi=300)
    h = np.arange(1, 8)
    for k, (name, v) in enumerate(tabs.items()):
        ax.plot(h, v, color=GR[k % 4], marker=MK[k % 6], ms=3, lw=0.9, ls=LS[k % 4], label=name)
    ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("Daily MAE (O$_2$ starts)")
    vals = np.concatenate(list(tabs.values()))
    ax.set_ylim(vals.min() * 0.93, vals.max() * 1.18)
    ax.legend(fontsize=5.4, ncol=2, loc="upper left", borderaxespad=0.1, columnspacing=0.8)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_reliability(curves, path):
    """curves: dict name -> (y, r) on test set."""
    fig, ax = plt.subplots(figsize=(3.45, 2.2), dpi=300)
    ax.plot([0, 0.8], [0, 0.8], color="#999999", lw=0.6, ls="--")
    for k, (name, (y, r)) in enumerate(curves.items()):
        qs = np.quantile(r, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(qs, r, side="right") - 1, 0, 9)
        xs = [r[b == j].mean() for j in range(10)]; ys = [y[b == j].mean() for j in range(10)]
        ax.plot(xs, ys, color=GR[k % 4], marker=MK[k], ms=3, lw=0.9, ls=LS[k % 4], label=name)
    ax.set_xlabel("Predicted 72-h risk (decile mean)"); ax.set_ylabel("Observed rate")
    ax.set_xlim(0, 0.82); ax.set_ylim(0, 0.82)
    ax.legend(fontsize=5.6, loc="upper left", borderaxespad=0.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_tierb(tb, path):
    weeks = tb["weeks"]; Y = tb["Y"]; pr = tb["preds"]; t0 = tb["t_first"]
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 2.6), dpi=300, sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.42))
    names = ["Ontario", "Quebec", "British Columbia", "Alberta"]
    for k, ax in enumerate(axs.ravel()):
        sl = slice(t0 - 10, len(weeks))
        ax.plot(weeks[sl], Y[k, sl], color="black", lw=0.9, label="Observed")
        f = np.full(len(weeks), np.nan); f2 = np.full(len(weeks), np.nan)
        f[t0 + 3:] = pr["AR + wastewater ridge"][k, t0:len(weeks) - 3, 2]
        f2[t0 + 3:] = pr["AR ridge"][k, t0:len(weeks) - 3, 2]
        ax.plot(weeks, f, color="#555555", lw=0.8, ls="--", label="+ wastewater")
        ax.plot(weeks, f2, color="#9A9A9A", lw=0.8, ls=":", label="AR only")
        ax.set_title(names[k], fontsize=6.4, loc="left", pad=1.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
        ax.set_ylim(0, max(np.nanmax(Y[k, sl]), np.nanmax(f), np.nanmax(f2)) * 1.1)
        ax.tick_params(labelsize=5.6)
    axs[0, 0].set_ylabel("Weekly admissions", fontsize=6); axs[1, 0].set_ylabel("Weekly admissions", fontsize=6)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.6, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.97), columnspacing=1.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_loso(fed, loso, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), dpi=300, gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.6))
    ax = axs[0]
    lab = ["Pooled", "FedAvg", "FedProx", "Local"]
    v = fed.AUROC.values; ws = fed.worst_site.values
    xs = np.arange(len(v))
    ax.bar(xs, v, 0.6, color=["#555555", "#8C8C8C", "#B5B5B5", "#DDDDDD"][:len(v)], edgecolor="black", lw=0.5)
    ax.scatter(xs, ws, marker="_", s=60, color="black", zorder=3, label="Worst site")
    ax.set_xticks(xs); ax.set_xticklabels(lab[:len(v)], rotation=30, fontsize=5.8)
    ax.set_ylim(min(ws.min(), v.min()) - 0.02, v.max() + 0.03); ax.set_ylabel("Test AUROC")
    ax.legend(fontsize=5.4, loc="upper right", borderaxespad=0.0, handletextpad=0.2)
    ax.set_title("(a) Federation", fontsize=7)
    ax = axs[1]
    if len(loso):
        xs = np.arange(len(loso)); w = 0.36
        ax.bar(xs - w / 2, loso.auc_nopress, w, color="#BDBDBD", edgecolor="black", lw=0.5, label="No pressure")
        ax.bar(xs + w / 2, loso.auc, w, color="#555555", edgecolor="black", lw=0.5, label="COHERE-Resp")
        ax.set_xticks(xs); ax.set_xticklabels(loso.city, rotation=30, fontsize=5.8)
        lo = min(loso.auc.min(), loso.auc_nopress.min()); hi = max(loso.auc.max(), loso.auc_nopress.max())
        ax.set_ylim(lo - 0.02, hi + 0.025)
        ax.legend(fontsize=5.4, loc="upper left", ncol=2, borderaxespad=0.0, columnspacing=0.6)
    ax.set_title("(b) Leave-city-out", fontsize=7)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_tierc(res, path):
    """Tier C: pooled AUROC per model (mean and SD over seeds) and the local-model transfer matrix."""
    order = [("Logistic regression (hard)", "Logistic reg."), ("LightGBM (hard)", "LightGBM"), ("GRU-D (hard)", "GRU-D"),
             ("Transformer (deterministic)", "TF, determ."), ("Transformer (hard)", "TF, thresh."),
             ("COHERE (soft PMA)", "COHERE-Resp"), ("COHERE federated", "COHERE fed."), ("Oracle linkage", "Oracle links")]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9), dpi=300, gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.45))
    ax = axs[0]
    p = res[res.split == "pooled"].groupby("model").AUROC.agg(["mean", "std"])
    xs = np.arange(len(order))
    m = [p.loc[k, "mean"] for k, _ in order]; s = [p.loc[k, "std"] for k, _ in order]
    cols = ["#DDDDDD", "#DDDDDD", "#B5B5B5", "#8C8C8C", "#8C8C8C", "#444444", "#444444", "#FFFFFF"]
    ax.bar(xs, m, 0.65, yerr=s, color=cols, edgecolor="black", lw=0.5, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in order], rotation=40, ha="right", fontsize=5.6)
    ax.set_ylim(0.55, max(m) + 0.03); ax.set_ylabel("Pooled test AUROC")
    ax.set_title("(a) Sepsis within 24 h", fontsize=7)
    ax = axs[1]
    t = res[res.split == "transfer"].groupby("model").AUROC.mean()
    M = np.array([[t["local A on A"], t["local A on B"]], [t["local B on A"], t["local B on B"]]])
    im = ax.imshow(M, cmap="Greys", vmin=0.5, vmax=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["A", "B"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["A", "B"])
    ax.set_xlabel("Tested on"); ax.set_ylabel("Trained on")
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.set_title("(b) Local transfer", fontsize=7)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)
