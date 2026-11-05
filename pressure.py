"""Epidemic-pressure features from real wastewater surveillance.

SARS-CoV-2: PHAC National Wastewater Surveillance (N2 copies/mL), one sewershed per site,
reproduced by the COVID-19 Canada Open Data Working Group (CovidTimelineCanada, CC BY 4.0).
Influenza A+B and RSV: Ottawa ROPEC PMMoV-normalised series (PHESD, Delatolla Lab, CC BY 4.0),
shared by all sites with a site-specific timing offset because PHAC multi-pathogen site series
are not part of the open repository.
"""
import numpy as np
import pandas as pd
from config import RAW, SITES, START, END, REPORT_DELAY, VARIANT_MID

DAYS = pd.date_range(START, END, freq="D")
SITE_FLU_SHIFT = [0, 0, 7, 7, -10, -10, 3, 10, -5, 2]   # days; west coast earlier, east later


def _daily_log_series(dates, values, days, max_gap=14):
    """Log-linear interpolation of irregular samples onto a daily grid (no extrapolation).
    Non-detects (zero or below the 5th percentile of positive results) are set to that floor
    before the log transform, the usual substitution for values under the detection limit."""
    v = np.asarray(values, float)
    pos = v[v > 0]
    floor = np.percentile(pos, 5) if len(pos) else 1e-6
    s = pd.Series(np.log10(np.maximum(v, floor)), index=pd.to_datetime(dates))
    s = s.groupby(level=0).mean().sort_index()
    full = s.reindex(s.index.union(days)).interpolate(method="time", limit_area="inside")
    full = full.reindex(days)
    # mark days further than max_gap from any sample as missing
    idx = s.index.values
    pos = np.searchsorted(idx, days.values)
    prev_gap = np.full(len(days), 1e9)
    ok = pos > 0
    prev_gap[ok] = (days.values[ok] - idx[pos[ok] - 1]) / np.timedelta64(1, "D")
    exact = np.isin(days.values, idx)
    prev_gap[exact] = 0
    full[(prev_gap > max_gap)] = np.nan
    return full.values, prev_gap


def variant_prevalence(days):
    t = (days - pd.Timestamp("2021-01-01")).days.values.astype(float)
    mids = [(pd.Timestamp(m) - pd.Timestamp("2021-01-01")).days for m in VARIANT_MID]
    cum = [1.0 / (1.0 + np.exp(-(t - m) / 12.0)) for m in mids]   # share past each transition
    prev = np.zeros((len(days), len(mids) + 1))
    prev[:, 0] = 1 - cum[0]
    for k in range(1, len(mids)):
        prev[:, k] = cum[k - 1] - cum[k]
    prev[:, -1] = cum[-1]
    return np.clip(prev, 0, 1)


def load_raw():
    ww = pd.read_csv(f"{RAW}/phac_ww.csv")
    ott = pd.read_csv(f"{RAW}/ottawa_ww.csv")
    for c in ["InfA_copies_per_pep_copies_avg", "InfB_copies_per_pep_copies_avg", "RSV_copies_per_pep_copies_avg"]:
        ott[c] = pd.to_numeric(ott[c], errors="coerce")      # "Not tested" -> NaN
    return ww, ott


def build(days=DAYS):
    """Returns dict with
    truth[s]: array (D,3) log-concentration (sars2, flu, rsv) without reporting delay,
    feats: array (S,D,16) features available to models on each day (with reporting delay),
    """
    ww, ott = load_raw()
    ext = pd.date_range(days[0] - pd.Timedelta(days=60), days[-1], freq="D")
    flu_raw = ott["InfA_copies_per_pep_copies_avg"].fillna(0) + ott["InfB_copies_per_pep_copies_avg"].fillna(0)
    flu_ok = ott["InfA_copies_per_pep_copies_avg"].notna() | ott["InfB_copies_per_pep_copies_avg"].notna()
    flu, flu_gap = _daily_log_series(ott.sampleDate[flu_ok], flu_raw[flu_ok], ext)
    r_ok = ott["RSV_copies_per_pep_copies_avg"].notna()
    rsv, rsv_gap = _daily_log_series(ott.sampleDate[r_ok], ott["RSV_copies_per_pep_copies_avg"][r_ok], ext)
    vp = variant_prevalence(days)
    S, D = len(SITES), len(days)
    truth = np.full((S, D, 3), np.nan)
    feats = np.zeros((S, D, 16), np.float32)
    for si, site in enumerate(SITES):
        x = ww[ww.sub_region_2 == site]
        sc, sgap = _daily_log_series(x.date, x.value, ext)
        sh = SITE_FLU_SHIFT[si]
        f_s = np.roll(flu, sh); r_s = np.roll(rsv, sh)
        if sh > 0:
            f_s[:sh] = np.nan; r_s[:sh] = np.nan
        elif sh < 0:
            f_s[sh:] = np.nan; r_s[sh:] = np.nan
        mat = np.stack([sc, f_s, r_s], 1)
        # normalise each pathogen by its site median over the study period
        med = np.nanmedian(mat[60:], axis=0)
        mat = mat - med
        truth[si] = mat[60:]
        # model-facing features: lagged by reporting delay
        lag = np.roll(mat, REPORT_DELAY, axis=0); lag[:REPORT_DELAY] = np.nan
        gap = np.roll(sgap, REPORT_DELAY) + REPORT_DELAY
        level = lag[60:]
        slope = level - np.roll(lag, 7, axis=0)[60:]
        mask = np.isfinite(level).astype(np.float32)
        slope = np.where(np.isfinite(slope), slope, 0.0)
        level = np.where(np.isfinite(level), level, 0.0)
        fresh = np.clip(gap[60:] / 14.0, 0, 2)
        feats[si] = np.concatenate([level, slope, mask, fresh[:, None], vp], 1)
    return {"days": days, "truth": truth, "feats": feats, "variant_prev": vp}


if __name__ == "__main__":
    P = build()
    print(P["feats"].shape, np.isnan(P["truth"]).mean(axis=(0, 1)))
    print(np.round(np.nanpercentile(P["truth"], [5, 50, 95], axis=(0, 1)), 2))
