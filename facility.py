"""Facility-level oxygen-demand forecasting and coherent patient-to-facility reconciliation.

Hierarchy per site and forecast origin d, horizon h:   Y = C + N
  C : next-day-h oxygen starts among patients already admitted and at risk at the end of day d
  N : oxygen starts among patients admitted after day d
Base forecasts: C from summed patient hazards (bottom-up), N from an admission forecaster times
the empirical post-admission oxygen-start profile, Y from a direct (top-down) regressor.
Reconciliation uses trace minimisation (MinT) with a shrunk error covariance from the
calibration window; reconciled C is pushed back to patients by proportional scaling.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from config import SITES, HORIZON_DAYS
import dataset

HZ = HORIZON_DAYS
S_MAT = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])     # rows: Y, C, N ; cols: C, N


def bin_probs(hz):
    """P(first event in day-bin k) from discrete-time hazards."""
    surv = np.cumprod(np.concatenate([np.ones((len(hz), 1)), 1 - hz[:, :-1]], 1), 1)
    return hz * surv


def retention_curves(lm, train_mask, max_j=6):
    """Kaplan-Meier probability of still being an inpatient at the start of day-bin k, by day of
    stay j at the landmark (discharge is a competing exit; oxygen starts are treated as
    censoring for this curve)."""
    j_all = np.minimum(lm.tau.values // 24, max_j)
    R = np.ones((max_j + 1, HZ))
    for j in range(max_j + 1):
        m = train_mask & (j_all == j)
        ev = (lm.ebin.values[m] == 0) & (lm.cens_days.values[m] <= HZ)      # discharged within window
        t = np.where(ev, lm.cens_days.values[m], np.where(lm.ebin.values[m] > 0, lm.ebin.values[m] - 1, HZ + 1))
        surv = 1.0
        for k in range(HZ):
            at_risk = (t >= k).sum()
            d_k = (ev & (t >= k) & (t < k + 1)).sum()
            R[j, k] = surv
            surv *= 1 - d_k / max(at_risk, 1)
    return R


def current_component(lm, hz, S, D, R=None):
    """Predicted and observed C[s, d, h]."""
    pk = bin_probs(hz)
    if R is not None:
        pk = pk * R[np.minimum(lm.tau.values // 24, R.shape[0] - 1)]
    Cp = np.zeros((S, D, HZ)); Co = np.zeros((S, D, HZ))
    # a landmark at the end of calendar day d carries lm.day = d + 1; bin k covers day d + k
    s, d = lm.site.values, lm.day.values - 1
    ok = (d >= 0) & (d < D)
    for k in range(HZ):
        np.add.at(Cp[:, :, k], (s[ok], d[ok]), pk[ok, k])
        np.add.at(Co[:, :, k], (s[ok], d[ok]), (lm.ebin.values[ok] == k + 1).astype(float))
    return Cp, Co


def post_admission_profile(pats, train_day_end):
    """q[k]: share of admitted patients whose oxygen start falls on day k after admission day."""
    p = pats[pats.day < train_day_end]
    q = np.zeros(HZ + 1)
    ev = p[p.event_h >= 0]
    k = ((ev.adm_h + ev.event_h) // 24 - ev.day).values
    for kk in range(HZ + 1):
        q[kk] = (k == kk).sum() / len(p)
    return q


def _lags(x, d, n=7):
    return np.stack([x[:, max(d - i, 0)] for i in range(n)], 1)


def design(fac, P, d, use_ww=True):
    """Per-site feature rows at origin day d."""
    A, O, Cn = fac["admissions"], fac["o2_start"], fac["census"]
    f = [np.log1p(_lags(A, d)), np.log1p(_lags(O, d)), np.log1p(Cn[:, d:d + 1]),
         np.log1p(A[:, max(d - 13, 0):d + 1].mean(1, keepdims=True))]
    if use_ww:
        pf = P["feats"][:, d]
        pf7 = P["feats"][:, max(d - 6, 0):d + 1].mean(1)
        f += [pf[:, :10], pf7[:, :6], pf[:, 10:]]
    f.append(np.eye(len(SITES)))
    return np.concatenate(f, 1)


def fit_direct(fac, P, days, target, use_ww=True, alpha=3.0, model="ridge"):
    """One regressor per horizon on log1p target; returns predictor function."""
    X = np.concatenate([design(fac, P, d, use_ww) for d in days])
    mods = []
    for h in range(1, HZ + 1):
        y = np.concatenate([np.log1p(target[:, d + h]) for d in days])
        if model == "ridge":
            m = Ridge(alpha=alpha).fit(X, y)
        else:
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=1).fit(X, y)
        mods.append(m)

    def pred(d):
        Xd = design(fac, P, d, use_ww)
        return np.stack([np.expm1(m.predict(Xd)).clip(0) for m in mods], 1)   # (S, HZ)
    return pred


def evaluate(C, R, P, train_end_day, cal_end_day, last_day):
    """C: simulation output (patients, facility); R: run dict with hz for every landmark."""
    pats, fac = C["patients"], C["facility"]
    lm = R["lm"]; hz = R["hz"]
    S, D = fac["o2_start"].shape
    O = fac["o2_start"]
    trm = lm.lm_abs.values < train_end_day * 24
    Rk = retention_curves(lm, trm)
    Cp, Co = current_component(lm, hz, S, D, Rk)
    q = post_admission_profile(pats, train_end_day)
    tr_days = np.arange(30, train_end_day - HZ)
    cal_days = np.arange(train_end_day, cal_end_day - HZ)
    te_days = np.arange(cal_end_day, last_day - HZ)
    adm_f = fit_direct(fac, P, tr_days, fac["admissions"], use_ww=True)
    adm_f_noww = fit_direct(fac, P, tr_days, fac["admissions"], use_ww=False)
    td_ww = fit_direct(fac, P, tr_days, O, use_ww=True)
    td_noww = fit_direct(fac, P, tr_days, O, use_ww=False)
    td_gbm = fit_direct(fac, P, tr_days, O, use_ww=True, model="gbm")

    def comps(d, adm_model=adm_f):
        Ahat = adm_model(d)                                      # (S, HZ) admissions on d+1..d+HZ
        N = np.zeros((S, HZ))
        for h in range(1, HZ + 1):
            for u in range(1, h + 1):
                N[:, h - 1] += Ahat[:, u - 1] * q[h - u]
        return Cp[:, d], N

    def collect(days):
        rows = []
        for d in days:
            c, n = comps(d)
            c0, n0 = comps(d, adm_f_noww)
            y = np.stack([O[:, d + h] for h in range(1, HZ + 1)], 1)
            rows.append(dict(d=d, y=y, c_obs=Co[:, d], c=c, n=n, bu=c + n, bu_noww=c0 + n0,
                             td=td_ww(d), td_noww=td_noww(d), td_gbm=td_gbm(d),
                             naive=np.repeat(O[:, max(d - 6, 0):d + 1].mean(1, keepdims=True), HZ, 1)))
        return rows

    cal, te = collect(cal_days), collect(te_days)
    # MinT weights per horizon from calibration residuals of base forecasts (Y=td, C, N)
    G = {}
    for h in range(HZ):
        E = []
        for r in cal:
            ey = r["y"][:, h] - r["td"][:, h]
            ec = r["c_obs"][:, h] - r["c"][:, h]
            en = (r["y"][:, h] - r["c_obs"][:, h]) - r["n"][:, h]
            E.append(np.stack([ey, ec, en], 1))
        E = np.concatenate(E)
        Wc = np.cov(E.T)
        lam = 0.3
        Wc = (1 - lam) * Wc + lam * np.diag(np.diag(Wc))
        Wi = np.linalg.inv(Wc)
        G[h] = np.linalg.solve(S_MAT.T @ Wi @ S_MAT, S_MAT.T @ Wi)      # 2x3
    for r in cal + te:
        rec = np.zeros((S, HZ)); cr = np.zeros((S, HZ))
        for h in range(HZ):
            base = np.stack([r["td"][:, h], r["c"][:, h], r["n"][:, h]], 1)
            bott = base @ G[h].T                                           # (S,2) reconciled C,N
            bott = np.clip(bott, 0, None)
            rec[:, h] = bott.sum(1); cr[:, h] = bott[:, 0]
        r["mint"] = rec; r["c_rec"] = cr
        # simple variance-weighted combination of BU and TD (no hierarchy) for comparison
    return dict(cal=cal, te=te, q=q, G=G)


def metrics(rows, key):
    y = np.stack([r["y"] for r in rows]); f = np.stack([r[key] for r in rows])     # (T,S,H)
    mae = np.abs(y - f).mean()
    rmse = np.sqrt(((y - f) ** 2).mean())
    y7, f7 = y.sum(2), f.sum(2)
    mape7 = (np.abs(y7 - f7) / np.maximum(y7, 1)).mean() * 100
    wape7 = np.abs(y7 - f7).sum() / y7.sum() * 100
    return dict(MAE=mae, RMSE=rmse, MAPE7=mape7, WAPE7=wape7, MAE_h=np.abs(y - f).mean((0, 1)))


def surge_lead(rows, key, thr):
    """Mean lead (days) with which the forecast 7-day total first exceeds the site surge
    threshold, relative to the observed 7-day total crossing it."""
    y7 = np.stack([r["y"].sum(1) for r in rows]); f7 = np.stack([r[key].sum(1) for r in rows])
    leads, missed, false_al = [], 0, 0
    T, S = y7.shape
    for s in range(S):
        obs = y7[:, s] >= thr[s]; fc = f7[:, s] >= thr[s]
        onsets = [t for t in range(1, T) if obs[t] and not obs[t - 1]]
        for t in onsets:
            prior = [u for u in range(max(0, t - 21), t + 1) if fc[u]]
            if prior:
                # earliest forecast crossing in the contiguous alarm run ending at/after t
                u = t
                while u - 1 >= max(0, t - 21) and fc[u - 1]:
                    u -= 1
                leads.append(t - u if fc[t] else t - max(prior))
            else:
                missed += 1
        false_al += sum(1 for t in range(1, T) if fc[t] and not fc[t - 1] and not obs[t:min(T, t + 14)].any())
    return dict(lead=np.mean(leads) if leads else np.nan, n_onsets=len(leads) + missed, missed=missed,
                false_alarms=false_al)
