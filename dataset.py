"""Landmark datasets: token sequences assembled from linked fragments under a linkage policy."""
import numpy as np
import pandas as pd
from config import MAX_TOKENS, LOOKBACK_H, HORIZON_DAYS, START, TRAIN_END, CAL_END, PRESSURE_WINDOW
from simulate import F

LOG_FEATS = [5, 8]          # CRP, D-dimer modelled on log scale


def _split_hours():
    t0 = pd.Timestamp(START)
    return ((pd.Timestamp(TRAIN_END) - t0).days + 1) * 24, ((pd.Timestamp(CAL_END) - t0).days + 1) * 24


def landmarks(pats, max_day=9):
    """End-of-day landmarks for every at-risk inpatient (days 0..max_day of the stay)."""
    rows = []
    for r in pats.itertuples(index=False):
        end_stay = r.adm_h + (r.event_h if r.event_h >= 0 else r.los)
        for k in range(max_day + 1):
            lm_abs = (r.adm_h // 24 + 1 + k) * 24          # end of calendar day
            tau = lm_abs - r.adm_h
            if lm_abs >= end_stay:
                break
            if r.event_h >= 0 and r.event_h <= tau:
                break
            if r.event_h >= 0:
                dt = r.event_h - tau
                ebin = int(np.ceil(dt / 24.0))
                ebin = ebin if ebin <= HORIZON_DAYS else 0
            else:
                ebin = 0
            cens = (r.los - tau) / 24.0 if r.event_h < 0 else HORIZON_DAYS + 1
            rows.append((r.pid, r.site, lm_abs, tau, lm_abs // 24, ebin, min(cens, HORIZON_DAYS + 1),
                         int(r.event_h >= 0 and tau < r.event_h <= tau + 72)))
    return pd.DataFrame(rows, columns=["pid", "site", "lm_abs", "tau", "day", "ebin", "cens_days", "y72"])


def link_weights(L, mode, pats, frags):
    """Returns dict anchor_pid -> list of (fid, weight) for non-ward fragments."""
    pairs, f = L["pairs"], frags
    j, i = pairs[:, 0], pairs[:, 1]
    anc_pid = f.pid.values[i]
    if mode == "oracle":
        own = f[(f.src != "ward") & (f.pid >= 0)]
        out = {}
        for fid, p in zip(own.fid.values, own.pid.values):
            out.setdefault(p, []).append((fid, 1.0))
        return out
    if mode == "soft":
        w = L["pi"]; keep = w > 0.02
    elif mode == "hard":
        from linkage import hard_assign
        w = hard_assign(pairs, L["pi"]); keep = w > 0
    elif mode == "det":
        w = L["det"]; keep = w > 0
    elif mode == "fs":
        from linkage import hard_assign
        w = hard_assign(pairs, L["fs"]); keep = w > 0
    elif mode == "none":
        return {}
    else:
        raise ValueError(mode)
    out = {}
    for a, fj, ww in zip(anc_pid[keep], f.fid.values[j[keep]], w[keep]):
        out.setdefault(a, []).append((fj, float(ww)))
    return out


class ObsIndex:
    def __init__(self, C):
        meta = C["obs_meta"]
        order = np.argsort(meta[:, 0], kind="stable")
        self.meta = meta[order]; self.vals = C["obs_vals"][order]
        fids = self.meta[:, 0].astype(int)
        self.start = np.searchsorted(fids, np.arange(fids.max() + 2))

    def get(self, fid):
        a, b = self.start[fid], self.start[fid + 1]
        return self.meta[a:b], self.vals[a:b]


def build(C, L, mode, lm=None, stats=None):
    pats, frags = C["patients"], L["frags"]
    if lm is None:
        lm = landmarks(pats)
    ward_fid = frags[frags.src == "ward"].set_index("pid").fid
    links = link_weights(L, mode, pats, frags)
    oi = ObsIndex(C)
    N = len(lm)
    X = np.full((N, MAX_TOKENS, F), np.nan, np.float16)
    T = np.zeros((N, MAX_TOKENS), np.float32)
    S = np.zeros((N, MAX_TOKENS), np.int8)
    W = np.zeros((N, MAX_TOKENS), np.float32)
    PAD = np.ones((N, MAX_TOKENS), bool)
    adm = pats.set_index("pid").adm_h
    for n, r in enumerate(lm.itertuples(index=False)):
        a0 = adm[r.pid]
        lo, hi = a0 - LOOKBACK_H, r.lm_abs
        parts = []
        m, v = oi.get(ward_fid[r.pid])
        sel = (m[:, 1] <= hi) & (m[:, 1] >= r.lm_abs - 72)
        parts.append((m[sel], v[sel], np.ones(sel.sum())))
        for fid, w in links.get(r.pid, []):
            m, v = oi.get(fid)
            sel = (m[:, 1] <= hi) & (m[:, 1] >= lo - 72)
            if sel.any():
                parts.append((m[sel], v[sel], np.full(sel.sum(), w)))
        mm = np.concatenate([p[0] for p in parts]); vv = np.concatenate([p[1] for p in parts])
        ww = np.concatenate([p[2] for p in parts])
        if len(mm) > MAX_TOKENS:
            ward = mm[:, 2] == 0
            keep_other = np.where(~ward)[0]
            wi = np.where(ward)[0]
            wi = wi[np.argsort(mm[wi, 1])][-(MAX_TOKENS - min(len(keep_other), MAX_TOKENS // 2)):]
            idx = np.concatenate([keep_other[:MAX_TOKENS // 2], wi])
            mm, vv, ww = mm[idx], vv[idx], ww[idx]
        o = np.argsort(mm[:, 1]); mm, vv, ww = mm[o], vv[o], ww[o]
        k = len(mm)
        X[n, :k] = vv; T[n, :k] = (mm[:, 1] - r.lm_abs) / 24.0; S[n, :k] = mm[:, 2]; W[n, :k] = ww
        PAD[n, :k] = False
    Xf = X.astype(np.float32)
    Xf[..., LOG_FEATS] = np.log(np.clip(Xf[..., LOG_FEATS], 1e-3, None))
    if stats is None:
        tr_h, _ = _split_hours()
        trm = (lm.lm_abs.values < tr_h)
        mu = np.nanmean(Xf[trm].reshape(-1, F), 0); sd = np.nanstd(Xf[trm].reshape(-1, F), 0) + 1e-6
        mu[14:] = 0; sd[14:] = 1                       # one-hot / counts left on native scale
        sd[17] = 2.0
        stats = (mu, sd)
    Xf = (Xf - stats[0]) / stats[1]
    st = pats.set_index("pid").loc[lm.pid, ["age", "sex"]].values.astype(np.float32)
    static = np.stack([(st[:, 0] - 65) / 15, st[:, 1], lm.tau.values / 72.0], 1).astype(np.float32)
    return dict(X=Xf.astype(np.float16), T=T, S=S, W=W, PAD=PAD, static=static, lm=lm.reset_index(drop=True),
                stats=stats, mode=mode)


def splits(lm):
    tr_h, cal_h = _split_hours()
    t = lm.lm_abs.values
    return t < tr_h, (t >= tr_h) & (t < cal_h), t >= cal_h


def pressure_tokens(P, site, day):
    """(n, W, 16) pressure history ending at the landmark day."""
    feats = P["feats"]
    idx = day[:, None] - np.arange(PRESSURE_WINDOW)[::-1][None, :]
    idx = np.clip(idx, 0, feats.shape[1] - 1)
    return feats[site[:, None], idx]
