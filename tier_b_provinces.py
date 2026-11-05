"""Tier B: real weekly COVID-19 hospital admissions in six Canadian provinces versus real PHAC
wastewater (province mean of site log-concentrations).  Rolling-origin, direct multi-horizon
forecasts (1-3 weeks) with and without the wastewater-pressure pathway; split, weighted and
adaptive conformal intervals; Diebold-Mariano tests."""
import pickle
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import conformal
from config import WORK

PROV = ["ON", "QC", "BC", "AB"]   # MB, NS excluded: zero-report weeks and lump reporting
H = 3


def load():
    a = pd.read_csv("hosp_admissions_pt.csv", parse_dates=["date"])
    ww = pd.read_csv("phac_ww.csv", parse_dates=["date"])
    weeks = pd.date_range("2021-06-06", "2023-11-26", freq="W-SUN")
    Y = np.zeros((len(PROV), len(weeks))); Wl = np.full((len(PROV), len(weeks)), np.nan)
    for k, p in enumerate(PROV):
        s = a[a.region == p].set_index("date").value_daily.clip(lower=0).resample("W-SUN").sum()
        Y[k] = s.reindex(weeks).fillna(0).values
        w = ww[ww.region == p].copy()
        w["lv"] = np.log10(w.value.clip(lower=0) + 1)
        w["date"] = w.date + pd.Timedelta(days=3)               # reporting delay
        site_week = w.set_index("date").groupby("sub_region_2").lv.resample("W-SUN").mean().unstack(0)
        Wl[k] = site_week.mean(axis=1).reindex(weeks).interpolate(limit=2).values
    return weeks, Y, Wl


def design(Y, Wl, t, ww=True):
    rows = []
    for k in range(Y.shape[0]):
        f = [np.log1p(Y[k, t - i]) for i in range(4)]
        if ww:
            w0 = Wl[k, t] if np.isfinite(Wl[k, t]) else np.nanmean(Wl[k, max(t - 4, 0):t + 1])
            w1 = Wl[k, t - 1] if np.isfinite(Wl[k, t - 1]) else w0
            w2 = Wl[k, t - 2] if np.isfinite(Wl[k, t - 2]) else w1
            f += [w0, w0 - w1, w1 - w2]
        f += list(np.eye(Y.shape[0])[k])
        rows.append(f)
    return np.nan_to_num(np.array(rows))


def run():
    weeks, Y, Wl = load()
    T = len(weeks)
    t_first = np.searchsorted(weeks, pd.Timestamp("2022-09-04"))
    models = {"Persistence": None, "AR ridge": dict(ww=False, kind="ridge"),
              "AR + wastewater ridge": dict(ww=True, kind="ridge"),
              "AR LightGBM": dict(ww=False, kind="gbm"), "AR + wastewater LightGBM": dict(ww=True, kind="gbm")}
    preds = {m: np.full((len(PROV), T, H), np.nan) for m in models}
    for t in range(t_first, T - 1):
        if (t - t_first) % 4 == 0:                      # refit every 4 weeks on expanding window
            fitted = {}
            for m, cfg in models.items():
                if cfg is None:
                    continue
                hs = []
                for h in range(1, H + 1):
                    tt = np.arange(4, t - h + 1)
                    X = np.concatenate([design(Y, Wl, u, cfg["ww"]) for u in tt])
                    y = np.concatenate([np.log1p(Y[:, u + h]) for u in tt])
                    if cfg["kind"] == "ridge":
                        hs.append(Ridge(alpha=1.0).fit(X, y))
                    else:
                        import lightgbm as lgb
                        hs.append(lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7,
                                                    min_child_samples=10, verbose=-1, n_jobs=1).fit(X, y))
                fitted[m] = hs
        for m, cfg in models.items():
            for h in range(1, H + 1):
                if t + h >= T:
                    continue
                if cfg is None:
                    preds[m][:, t, h - 1] = Y[:, t]
                else:
                    preds[m][:, t, h - 1] = np.expm1(fitted[m][h - 1].predict(design(Y, Wl, t, cfg["ww"]))).clip(0)
    res = []
    for m in models:
        for h in range(H):
            e, yy = [], []
            for t in range(t_first, T - 1 - h):
                e.append(preds[m][:, t, h] - Y[:, t + h + 1]); yy.append(Y[:, t + h + 1])
            e, yy = np.array(e), np.array(yy)
            res.append(dict(model=m, h=h + 1, MAE=np.abs(e).mean(), MAPE=(np.abs(e) / np.maximum(yy, 1)).mean() * 100,
                            RMSE=np.sqrt((e ** 2).mean())))
    res = pd.DataFrame(res)
    # Diebold-Mariano (absolute loss, HAC variance) wastewater vs no wastewater, pooled over provinces
    dm = []
    for h in range(H):
        for a, b in [("AR + wastewater ridge", "AR ridge"), ("AR + wastewater LightGBM", "AR LightGBM")]:
            dl = []
            for t in range(t_first, T - 1 - h):
                dl.append(np.abs(preds[a][:, t, h] - Y[:, t + h + 1]) - np.abs(preds[b][:, t, h] - Y[:, t + h + 1]))
            dl = np.array(dl)                              # (time, prov)
            d = dl.mean(1)
            n = len(d); g0 = np.var(d, ddof=0)
            lagmax = h + 1
            gam = sum(2 * (1 - l / (lagmax + 1)) * np.cov(d[l:], d[:-l], ddof=0)[0, 1] for l in range(1, lagmax + 1))
            stat = d.mean() / np.sqrt((g0 + gam) / n)
            dm.append(dict(h=h + 1, pair=f"{a} vs {b}", mean_diff=d.mean(), DM=stat, p=2 * (1 - stats.norm.cdf(abs(stat)))))
    dm = pd.DataFrame(dm)
    # conformal intervals for the wastewater ridge forecaster, 1-week horizon, scores on log scale
    cov = []
    m = "AR + wastewater ridge"
    for h in range(H):
        sc_hist, Z_hist, out = [], [], []
        for t in range(t_first, T - 1 - h):
            f = preds[m][:, t, h]; y = Y[:, t + h + 1]
            s = np.abs(np.log1p(y) - np.log1p(f))
            z = np.stack([np.nan_to_num(Wl[:, t]), np.log1p(Y[:, t])], 1)
            if len(sc_hist) >= 6 * len(PROV):
                sc = np.concatenate(sc_hist[-26:]); Zc = np.concatenate(Z_hist[-26:])
                n = len(sc)
                q = np.quantile(sc, min(1, np.ceil((n + 1) * 0.9) / n), method="higher")
                wc, wt = conformal.shift_weights(Zc, z, C=0.1, clip=(0.2, 5))
                qw = np.array([conformal.weighted_quantile(sc, wc, w_, 0.9) for w_ in wt])
                out.append(dict(split=(s <= q).mean(), w=(s <= qw).mean(), width=np.mean(np.expm1(np.log1p(f) + q) - np.expm1(np.log1p(f) - q)),
                                width_w=np.mean(np.expm1(np.log1p(f) + qw) - np.expm1(np.log1p(f) - qw))))
            sc_hist.append(s); Z_hist.append(z)
        o = pd.DataFrame(out)
        cov.append(dict(h=h + 1, split_cov=o.split.mean(), weighted_cov=o.w.mean(), split_width=o.width.mean(),
                        weighted_width=o.width_w.mean()))
    cov = pd.DataFrame(cov)
    out = dict(res=res, dm=dm, cov=cov, preds=preds, Y=Y, Wl=Wl, weeks=weeks, t_first=t_first)
    pickle.dump(out, open(f"{WORK}/tierB_results.pkl", "wb"))
    print(res.round(2)); print(dm.round(4)); print(cov.round(3))
    # lead correlation: wastewater vs admissions
    for k, p in enumerate(PROV):
        cc = [np.corrcoef(np.nan_to_num(Wl[k, 10:T - 6]), np.log1p(Y[k, 10 + l:T - 6 + l]))[0, 1] for l in range(0, 5)]
        print(p, "corr by lead (weeks 0..4):", np.round(cc, 3))
    return out


if __name__ == "__main__":
    run()
