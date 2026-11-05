"""Aggregate all saved runs into results (JSON) and publication figures."""
import json, os, pickle, glob, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import facility, conformal, facility_run
from config import WORK, SITES, START, TRAIN_END, CAL_END

RUNS = f"{WORK}/runs"
_prep_cache = {}


def prep(tag):
    if tag not in _prep_cache:
        _prep_cache[tag] = pickle.load(open(f"{WORK}/prep_{tag}.pkl", "rb"))
    return _prep_cache[tag]


def load_run(tag, name, seed=0, ho=None):
    p = f"{RUNS}/{tag}__{name}__{seed}" + (f"__ho{ho}" if ho else "") + ".pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def ece(y, r, bins=10):
    b = np.clip((r * bins).astype(int), 0, bins - 1)
    return sum(abs(y[b == k].mean() - r[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any())


def scores(y, r):
    return dict(AUROC=roc_auc_score(y, r), AUPRC=average_precision_score(y, r), Brier=np.mean((r - y) ** 2), ECE=ece(y, r))


def boot_ci(y, r, pid, n=1000, seed=0, r2=None):
    rng = np.random.default_rng(seed)
    up, inv = np.unique(pid, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(up))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(up), len(up))])
        a = roc_auc_score(y[idx], r[idx])
        vals.append(a - roc_auc_score(y[idx], r2[idx]) if r2 is not None else a)
    return np.percentile(vals, [2.5, 97.5])


def boot_ci_seeds(y, ra, rb, pid, n=300, seed=0):
    """Patient-cluster bootstrap CI of the seed-averaged AUROC difference."""
    rng = np.random.default_rng(seed)
    up, inv = np.unique(pid, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(up))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(up), len(up))])
        vals.append(np.mean([roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]) for a, b in zip(ra, rb)]))
    return np.percentile(vals, [2.5, 97.5])


def recal_ece(y, r, split):
    """ECE on the test period after Platt scaling fitted on the calibration window (same for every model)."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(r, 1e-6, 1 - 1e-6) / (1 - np.clip(r, 1e-6, 1 - 1e-6)))[:, None]
    lr = LogisticRegression(C=1e6).fit(z[split["ca"]], y[split["ca"]])
    return ece(y[split["te"]], lr.predict_proba(z[split["te"]])[:, 1])


def site_aurocs(run, lm):
    te = run["split"]["te"]; y = lm.y72.values; out = []
    for s in range(len(SITES)):
        m = te & (lm.site.values == s)
        out.append(roc_auc_score(y[m], run["risk"][m]))
    return np.array(out)


MAIN = [("lr", "Logistic regression"), ("lgbm", "LightGBM"), ("lgbm_ww", "LightGBM + wastewater"),
        ("lgbm_soft_ww", "LightGBM, $\\pi$-weighted + wastewater"), ("grud", "GRU-D"), ("grud_ww", "GRU-D + wastewater"),
        ("tf_hard", "Time-aware Transformer"), ("cohere", "COHERE-Resp (pooled)"), ("cohere_fed", "COHERE-Resp (federated)")]
TAGS = ["s0_c20", "s1_c20", "s2_c20"]


def main_table():
    rows, site_auc = [], {}
    for key, name in MAIN:
        per = []; sa = []
        for tag in TAGS:
            run = load_run(tag, key)
            if run is None:
                continue
            lm = prep(tag)["lm"]; te = run["split"]["te"]
            sc = scores(lm.y72.values[te], run["risk"][te]); sc["ECEr"] = recal_ece(lm.y72.values, run["risk"], run["split"])
            per.append(sc); sa.append(site_aurocs(run, lm))
        if not per:
            continue
        df = pd.DataFrame(per)
        rows.append(dict(key=key, model=name, n=len(per), **{f"{m}_mean": df[m].mean() for m in df},
                         **{f"{m}_sd": df[m].std(ddof=1) if len(df) > 1 else 0 for m in df}))
        site_auc[key] = np.concatenate(sa)
    T = pd.DataFrame(rows)
    # bootstrap CI on seed-0 cohort
    lm0 = prep("s0_c20")["lm"]
    ci = {}
    for key, _ in MAIN:
        run = load_run("s0_c20", key)
        if run is None:
            continue
        te = run["split"]["te"]
        ci[key] = boot_ci(lm0.y72.values[te], run["risk"][te], lm0.pid.values[te], n=300).tolist()
    # Wilcoxon COHERE vs each baseline over site x cohort AUROC
    tests = []
    for key, name in MAIN:
        if key == "cohere" or key not in site_auc or len(site_auc[key]) != len(site_auc["cohere"]):
            continue
        dlt = site_auc["cohere"] - site_auc[key]
        st = stats.wilcoxon(dlt)
        tests.append(dict(baseline=name, key=key, mean_diff=dlt.mean(), p=st.pvalue))
    tests = pd.DataFrame(tests)
    if len(tests):
        order = np.argsort(tests.p.values); m = len(tests); adj = np.zeros(m); run_max = 0
        for r, i in enumerate(order):
            run_max = max(run_max, min(1, tests.p.values[i] * (m - r))); adj[i] = run_max
        tests["p_holm"] = adj
    keys = [k for k in site_auc if len(site_auc[k]) == len(site_auc.get("cohere", []))]
    fr = stats.friedmanchisquare(*[site_auc[k] for k in keys]) if len(keys) >= 3 else None
    ranks = pd.DataFrame({k: site_auc[k] for k in keys}).rank(axis=1, ascending=False).mean().to_dict()
    return T, ci, tests, (fr.statistic, fr.pvalue) if fr else None, ranks


ABL = [("cohere", "Full model (soft links, membership bias, gated pressure)"),
       ("cohere_concat", "Pressure by concatenation instead of gated cross-attention"),
       ("cohere_nopress", "Without pressure"),
       ("cohere_hard", "Thresholded links, no membership bias"),
       ("tf_hard", "Thresholded links, no bias, no pressure"),
       ("cohere_fs", "Fellegi–Sunter links"), ("cohere_det", "Deterministic links"),
       ("oracle", "Oracle links (upper reference)")]


def ablation_table():
    rows = []
    lm = prep("s0_c20")["lm"]
    for key, name in ABL:
        per = []
        for sd in [0, 1, 2]:
            run = load_run("s0_c20", key, sd)
            if run is None:
                continue
            te = run["split"]["te"]
            sc = scores(lm.y72.values[te], run["risk"][te])
            try:
                df, ev = facility_run.run("s0_c20", key, sd, verbose=False)
                sc["WAPE7"] = float(df[df.key == "mint"].WAPE7.iloc[0]); sc["BU_WAPE7"] = float(df[df.key == "bu"].WAPE7.iloc[0])
            except Exception:
                pass
            per.append(sc)
        if per:
            df = pd.DataFrame(per)
            rows.append(dict(key=key, model=name, n=len(per), **{f"{m}_mean": df[m].mean() for m in df},
                             **{f"{m}_sd": df[m].std(ddof=1) if len(df) > 1 else 0 for m in df}))
    return pd.DataFrame(rows)


def corruption_sweep():
    rows = []
    for c in [0, 10, 20, 30, 40]:
        tag = f"s0_c{c}"
        if not os.path.exists(f"{WORK}/prep_{tag}.pkl"):
            continue
        P = prep(tag); lm = P["lm"]; lk = P["link_metrics"]
        row = dict(corruption=c / 100, f1_det=lk["deterministic"]["f1"], f1_fs=lk["fellegi_sunter"]["f1"],
                   f1_soft=lk["soft_argmax"]["f1"], rec_det=lk["deterministic"]["recall"], pi_ece=lk["pi_ece"])
        for key in ["cohere", "cohere_hard", "cohere_det", "lgbm", "lgbm_det", "lgbm_soft"]:
            run = load_run(tag, key)
            if run is not None:
                te = run["split"]["te"]
                sc = scores(lm.y72.values[te], run["risk"][te])
                row[f"auc_{key}"] = sc["AUROC"]; row[f"ece_{key}"] = sc["ECE"]
        rows.append(row)
    return pd.DataFrame(rows)


def pressure_subgroups():
    """COHERE-Resp vs no-pressure variant (s0, seeds 0-2) in clinically defined subgroups."""
    P = prep("s0_c20"); lm = P["lm"]; pats = P["C"]["patients"].set_index("pid")
    D = pickle.load(open(f"{WORK}/prep_s0_c20_soft.pkl", "rb"))
    known = ((D["S"] == 4) & (D["W"] > 0.5) & ~D["PAD"]).any(1)
    strain = pats.loc[lm.pid.values, "strain"].values > 0
    path = pats.loc[lm.pid.values, "pathogen"].values
    groups = {"All test landmarks": np.ones(len(lm), bool), "Pathogen result pending": ~known,
              "Pathogen result known": known, "First day of stay": lm.tau.values <= 24,
              "Ward under strain": strain, "Influenza": path == 1, "RSV": path == 2, "SARS-CoV-2": path == 0}
    rows = []
    for g, m in groups.items():
        d_auc, base = [], []
        for sd in [0, 1, 2]:
            a, b = load_run("s0_c20", "cohere", sd), load_run("s0_c20", "cohere_nopress", sd)
            if a is None or b is None:
                continue
            te = a["split"]["te"] & m
            y = lm.y72.values[te]
            if y.sum() < 10:
                continue
            ra, rb = roc_auc_score(y, a["risk"][te]), roc_auc_score(y, b["risk"][te])
            d_auc.append(ra - rb); base.append(rb)
        if d_auc:
            a0 = load_run("s0_c20", "cohere", 0)
            te = a0["split"]["te"] & m
            ci = boot_ci_seeds(lm.y72.values[te], [load_run("s0_c20", "cohere", sd)["risk"][te] for sd in [0, 1, 2]],
                               [load_run("s0_c20", "cohere_nopress", sd)["risk"][te] for sd in [0, 1, 2]], lm.pid.values[te])
            rows.append(dict(group=g, n=int((a0["split"]["te"] & m).sum()), base_auc=np.mean(base),
                             delta=np.mean(d_auc), delta_sd=np.std(d_auc, ddof=1), ci_lo=ci[0], ci_hi=ci[1]))
    sub = pd.DataFrame(rows)
    # null-pressure world
    null = []
    if os.path.exists(f"{WORK}/prep_s0_c20_null.pkl"):
        lmn = prep("s0_c20_null")["lm"]
        for sd in [0, 1]:
            a, b = load_run("s0_c20_null", "cohere", sd), load_run("s0_c20_null", "cohere_nopress", sd)
            if a is None or b is None:
                continue
            te = a["split"]["te"]
            null.append(dict(seed=sd, auc_press=roc_auc_score(lmn.y72.values[te], a["risk"][te]),
                             auc_nopress=roc_auc_score(lmn.y72.values[te], b["risk"][te]),
                             gate=a["extra"].get("gate_mean_test", np.nan)))
    gates = [load_run("s0_c20", "cohere", sd)["extra"].get("gate_mean_test", np.nan) for sd in [0, 1, 2] if load_run("s0_c20", "cohere", sd)]
    return sub, pd.DataFrame(null), gates


def day(s):
    return (pd.Timestamp(s) - pd.Timestamp(START)).days


def facility_all():
    out = {}
    for tag in TAGS:
        if load_run(tag, "cohere") is None:
            continue
        df, ev = facility_run.run(tag, "cohere", 0, verbose=False)
        out[tag] = (df, ev)
    return out


def dm_test(e1, e2, h=1):
    d = (np.abs(e1) - np.abs(e2)).mean(1)          # average over sites per origin
    n = len(d); g0 = np.var(d)
    gam = sum(2 * (1 - l / (h + 1)) * np.cov(d[l:], d[:-l], ddof=0)[0, 1] for l in range(1, h + 1))
    st = d.mean() / np.sqrt((g0 + gam) / n)
    return st, 2 * (1 - stats.norm.cdf(abs(st)))


def surge_lead(ev, tag, key):
    fac = prep(tag)["C"]["facility"]; O = fac["o2_start"]
    tr_end = day(TRAIN_END) + 1
    trailing = np.array([[O[s, max(0, t - 6):t + 1].sum() for t in range(O.shape[1])] for s in range(O.shape[0])])
    thr = np.percentile(trailing[:, 30:tr_end], 80, axis=1)
    te = ev["te"]; origins = np.array([r["d"] for r in te])
    f7 = np.stack([r[key].sum(1) for r in te], 1)           # (S, T)
    leads, missed, false_al, n_ep = [], 0, 0, 0
    for s in range(O.shape[0]):
        tr = trailing[s, origins] >= thr[s]; fc = f7[s] >= thr[s]
        onsets = [k for k in range(1, len(origins)) if tr[k] and not tr[k - 1]]
        n_ep += len(onsets)
        for k in onsets:
            win = [u for u in range(max(0, k - 14), k + 1) if fc[u]]
            if not win:
                missed += 1; continue
            u = k
            while u - 1 >= max(0, k - 14) and fc[u - 1]:
                u -= 1
            leads.append(k - u if fc[k] else k - max(win))
        for k in range(1, len(origins)):
            if fc[k] and not fc[k - 1] and not tr[k:min(len(origins), k + 15)].any():
                false_al += 1
    return dict(lead=float(np.mean(leads)) if leads else float("nan"), episodes=n_ep, detected=len(leads), missed=missed, false_alarms=false_al)


def facility_conformal(ev, tag, alpha=0.1, key="mint"):
    P = prep(tag)["pressure"]
    cal, te = ev["cal"], ev["te"]
    yc = np.concatenate([r["y"].sum(1) for r in cal]); fc = np.concatenate([r[key].sum(1) for r in cal])
    yt = np.concatenate([r["y"].sum(1) for r in te]); ft = np.concatenate([r[key].sum(1) for r in te])
    Zc = np.concatenate([P["feats"][:, r["d"], :10] for r in cal]); Zt = np.concatenate([P["feats"][:, r["d"], :10] for r in te])
    sc = np.abs(yc - fc)
    cov_s, w_s, _, _ = conformal.interval_coverage(yt, ft, sc, alpha)
    wc, wt = conformal.shift_weights(Zc, Zt)
    cov_w, w_w, _, _ = conformal.interval_coverage(yt, ft, sc, alpha, wc, wt)
    # ACI per site, sequential over test origins
    S = len(SITES); covs, wids = [], []
    for s in range(S):
        ys = np.array([r["y"][s].sum() for r in te]); fs = np.array([r[key][s].sum() for r in te])
        scs = np.array([abs(r["y"][s].sum() - r[key][s].sum()) for r in cal])
        c, w = conformal.aci(ys, fs, scs, alpha, gamma=0.02); covs.append(c); wids.append(w)
    return dict(split_cov=cov_s, split_width=w_s, weighted_cov=cov_w, weighted_width=w_w,
                aci_cov=float(np.mean(covs)), aci_width=float(np.mean(wids)))


def patient_conformal(tag="s0_c20", key="cohere", alpha=0.1):
    from dataset import pressure_tokens
    P = prep(tag); lm = P["lm"]; run = load_run(tag, key)
    ca, te = run["split"]["ca"], run["split"]["te"]; y = lm.y72.values; r = run["risk"]
    out = {}
    s = conformal.mondrian_sets(r[ca], y[ca], r[te], alpha)
    out["split"] = conformal.set_metrics(s, y[te])
    Zall = P["pressure"]["feats"][lm.site.values, np.clip(lm.day.values, 0, P["pressure"]["feats"].shape[1] - 1), :10]
    Zall = np.concatenate([Zall, (lm.tau.values / 72.0)[:, None]], 1)
    wc, wt = conformal.shift_weights(Zall[ca], Zall[te])
    s = conformal.mondrian_sets(r[ca], y[ca], r[te], alpha, wc, wt)
    out["weighted"] = conformal.set_metrics(s, y[te])
    return out


def label_delay():
    rows = []
    lm = prep("s0_c20")["lm"]
    for key, name in [("cohere_delay", "Delayed labels dropped"), ("cohere_delay_coh", "Delayed labels + coherence loss"), ("cohere", "All labels available (reference)")]:
        per = []
        for sd in [0, 1, 2]:
            run = load_run("s0_c20", key, sd)
            if run is None:
                continue
            te = run["split"]["te"]
            sc = scores(lm.y72.values[te], run["risk"][te])
            # calibration-in-the-large of day-1 hazards against observed next-day starts (test)
            hz = run["hz"]
            sc["obs_exp_day1"] = (lm.ebin.values[te] == 1).sum() / hz[te, 0].sum()
            df, ev = facility_run.run("s0_c20", key, sd, verbose=False)
            sc["BU_WAPE7"] = float(df[df.key == "bu"].WAPE7.iloc[0]); sc["MINT_WAPE7"] = float(df[df.key == "mint"].WAPE7.iloc[0])
            per.append(sc)
        if per:
            df = pd.DataFrame(per)
            rows.append(dict(key=key, setting=name, n=len(per), **{f"{m}_mean": df[m].mean() for m in df},
                             **{f"{m}_sd": df[m].std(ddof=1) if len(df) > 1 else 0 for m in df}))
    return pd.DataFrame(rows)


def federated():
    lm = prep("s0_c20")["lm"]; rows = []
    for key, name in [("cohere", "Pooled (upper reference)"), ("cohere_fed", "FedAvg + secure aggregation"),
                      ("cohere_fedprox", "FedProx ($\\mu$ = 0.01) + secure aggregation"), ("local", "Local-only models")]:
        run = load_run("s0_c20", key)
        if run is None:
            continue
        te = run["split"]["te"]
        sc = scores(lm.y72.values[te], run["risk"][te])
        sa = site_aurocs(run, lm)
        rows.append(dict(key=key, setting=name, **sc, worst_site=sa.min(), train_s=run["extra"].get("train_s")))
    loso = []
    folds = {"0-1": "Toronto", "2-3": "Montreal", "4-5": "Vancouver", "6-8-9": "Prairies", "7": "Halifax"}
    for ho, city in folds.items():
        a, b = load_run("s0_c20", "cohere", 0, ho), load_run("s0_c20", "cohere_nopress", 0, ho)
        if a is None:
            continue
        te = a["split"]["te"]
        loso.append(dict(city=city, n=int(te.sum()), auc=roc_auc_score(lm.y72.values[te], a["risk"][te]),
                         auc_nopress=roc_auc_score(lm.y72.values[te], b["risk"][te]) if b else np.nan,
                         ece=ece(lm.y72.values[te], a["risk"][te])))
    return pd.DataFrame(rows), pd.DataFrame(loso)


def secure_agg_check():
    import torch
    from train import secure_sum
    from models import CohereNet
    vecs = [torch.cat([p.detach().reshape(-1) for p in CohereNet().state_dict().values()]) for _ in range(10)]
    t = time.time(); agg, masked = secure_sum(vecs); dt = time.time() - t
    true = torch.stack(vecs).sum(0)
    return dict(max_abs_err=float((agg - true).abs().max()), n_params=int(len(vecs[0])), seconds=dt,
                masked_corr=float(np.corrcoef(masked[0].numpy(), vecs[0].numpy())[0, 1]))


def cohort_stats(tag="s0_c20"):
    P = prep(tag); C = P["C"]; lm = P["lm"]; pats = C["patients"]
    tr, ca, te = __import__("dataset").splits(lm)
    lk = P["link_metrics"]
    import simulate, linkage
    Cs = simulate.simulate(int(tag[1]))
    fr = Cs["frags"].reset_index(drop=True)
    n_pairs = len(linkage.candidate_pairs(fr))
    return dict(n_frags=len(fr), orphan_pct=float((fr.pid < 0).mean() * 100), n_pairs=n_pairs, n_patients=len(pats), n_covid=int((pats.pathogen == 0).sum()), n_flu=int((pats.pathogen == 1).sum()),
                n_rsv=int((pats.pathogen == 2).sum()), n_landmarks=len(lm),
                pos72=float(lm.y72.mean()), n_train=int(tr.sum()), n_cal=int(ca.sum()), n_test=int(te.sum()),
                link=lk)


def build_all():
    import figs_results as FG, pressure
    os.makedirs("figs", exist_ok=True)
    R = {"figs": {}}
    import platform, torch, sklearn, lightgbm
    R["env"] = (f"{os.cpu_count()} CPU threads, no GPU; Python {platform.python_version()}, PyTorch {torch.__version__.split('+')[0]}, "
                f"scikit-learn {sklearn.__version__}, LightGBM {lightgbm.__version__}")
    R["tierA"] = cohort_stats()
    T, ci, tests, fr, ranks = main_table()
    R["main"] = dict(table=T, ci=ci, tests=tests, friedman=fr, ranks=ranks)
    R["ablation"] = ablation_table()
    R["sweep"] = corruption_sweep()
    sub, null, gates = pressure_subgroups()
    R["pressure"] = dict(sub=sub, null=null, gates=gates)
    fac = facility_all()
    ftabs = []
    for tag, (df, ev) in fac.items():
        df = df.copy(); df["tag"] = tag; ftabs.append(df)
    FT = pd.concat(ftabs)
    R["facility"] = dict(table=FT.groupby(["key", "method"], sort=False).mean(numeric_only=True).reset_index(),
                         table_sd=FT.groupby(["key", "method"], sort=False).std(numeric_only=True).reset_index())
    # Diebold-Mariano: reconciled vs each base, pooled over cohorts (per-cohort tests reported)
    dms = []
    for tag, (df, ev) in fac.items():
        te = ev["te"]
        y = np.stack([r["y"] for r in te]); 
        for key in ["td", "td_gbm", "bu", "naive", "td_noww"]:
            e1 = (np.stack([r["mint"] for r in te]) - y).sum(2); e2 = (np.stack([r[key] for r in te]) - y).sum(2)
            st, p = dm_test(e1, e2, h=7)
            dms.append(dict(tag=tag, vs=key, DM=st, p=p))
    R["facility"]["dm"] = pd.DataFrame(dms)
    te0 = fac["s0_c20"][1]["te"]
    c0 = np.stack([r["c"] for r in te0]); n0 = np.stack([r["n"] for r in te0])
    R["facility"]["share_current"] = (c0.sum((0, 1)) / (c0 + n0).sum((0, 1))).tolist()
    leads = []
    for tag, (df, ev) in fac.items():
        for key in ["mint", "bu", "td", "td_gbm", "naive"]:
            leads.append(dict(tag=tag, key=key, **surge_lead(ev, tag, key)))
    R["facility"]["surge"] = pd.DataFrame(leads)
    fconf = [dict(tag=tag, **facility_conformal(ev, tag)) for tag, (df, ev) in fac.items()]
    R["facility"]["conformal"] = pd.DataFrame(fconf)
    R["patient_conformal"] = patient_conformal()
    R["label_delay"] = label_delay()
    fedt, loso = federated()
    R["federated"] = dict(table=fedt, loso=loso, secure=secure_agg_check())
    tb = pickle.load(open(f"{WORK}/tierB_results.pkl", "rb"))
    R["tierB"] = dict(res=tb["res"], dm=tb["dm"], cov=tb["cov"], n_weeks=len(tb["weeks"]),
                      n_eval=len(tb["weeks"]) - tb["t_first"])
    if os.path.exists(f"{WORK}/tierC_results.pkl"):
        tc = pickle.load(open(f"{WORK}/tierC_results.pkl", "rb")); tcp = pickle.load(open(f"{WORK}/tierC_prep.pkl", "rb"))
        lmC = tcp["lm"]
        R["tierC"] = dict(res=tc["res"], link=tcp["link"], n_landmarks=len(lmC), pos=float(lmC.y.mean()),
                          n_patients=int(lmC.pid.nunique()), n_A=int(lmC[lmC.site == 0].pid.nunique()),
                          n_B=int(lmC[lmC.site == 1].pid.nunique()), conformal=tc.get("conformal"))
    # efficiency
    eff = []
    for key, name in MAIN:
        run = load_run("s0_c20", key)
        if run is not None:
            eff.append(dict(key=key, model=name, params=run["extra"].get("n_params", 0), train_s=run["extra"].get("train_s"),
                            ms=run["extra"].get("ms_per_patient", np.nan)))
    R["efficiency"] = pd.DataFrame(eff)
    R["sites"] = site_table()
    # figures
    P0 = prep("s0_c20")["pressure"]
    FG.fig_series(P0, "figs/fig2_pressure_series.png"); R["figs"]["series"] = "figs/fig2_pressure_series.png"
    if len(R["sweep"]):
        FG.fig_corruption(R["sweep"], "figs/fig3_corruption.png"); R["figs"]["corruption"] = "figs/fig3_corruption.png"
    FG.fig_pressure(sub, null, "figs/fig4_pressure.png"); R["figs"]["pressure"] = "figs/fig4_pressure.png"
    df0, ev0 = fac["s0_c20"]
    qs = facility_conformal(ev0, "s0_c20")
    cal = ev0["cal"]
    sc = np.concatenate([np.abs(r["y"].sum(1) - r["mint"].sum(1)) for r in cal]); n = len(sc)
    q = np.quantile(sc, min(1, np.ceil((n + 1) * 0.9) / n), method="higher")
    FG.fig_facility(ev0, P0, "figs/fig5_facility.png", conf=q); R["figs"]["facility"] = "figs/fig5_facility.png"
    hz = {}
    for key, name in [("naive", "7-day mean"), ("td", "Direct ridge + WW"), ("td_gbm", "Direct LightGBM + WW"),
                      ("bu", "Bottom-up"), ("mint", "COHERE-Resp (MinT)")]:
        hz[name] = R["facility"]["table"].set_index("key").loc[key, [f"MAE_h{h}" for h in range(1, 8)]].values.astype(float)
    FG.fig_horizon(hz, "figs/fig6_horizon.png"); R["figs"]["horizon"] = "figs/fig6_horizon.png"
    lm0 = prep("s0_c20")["lm"]; curves = {}
    for key, name in [("cohere", "COHERE-Resp"), ("lgbm", "LightGBM"), ("grud", "GRU-D"), ("cohere_delay", "Delayed labels, no coherence"), ("cohere_delay_coh", "Delayed labels + coherence")]:
        run = load_run("s0_c20", key)
        if run is not None:
            te = run["split"]["te"]; curves[name] = (lm0.y72.values[te], run["risk"][te])
    FG.fig_reliability(curves, "figs/fig7_reliability.png"); R["figs"]["reliability"] = "figs/fig7_reliability.png"
    if len(fedt):
        FG.fig_loso(fedt, loso, "figs/fig8_federated.png"); R["figs"]["federated"] = "figs/fig8_federated.png"
    FG.fig_tierb(tb, "figs/fig9_tierb.png"); R["figs"]["tierb"] = "figs/fig9_tierb.png"
    if "tierC" in R:
        FG.fig_tierc(R["tierC"]["res"], "figs/fig10_tierc.png"); R["figs"]["tierc"] = "figs/fig10_tierc.png"
    R["figs"]["arch"] = "figs/fig1_architecture.png"
    pickle.dump(R, open(f"{WORK}/R.pkl", "wb"))
    return R


if __name__ == "__main__":
    R = build_all()
    for k in ["main", "ablation", "sweep", "pressure", "facility", "label_delay", "federated", "patient_conformal"]:
        print("=====", k)
        v = R[k]
        if isinstance(v, dict):
            for kk, vv in v.items():
                print("--", kk); print(vv.round(4).to_string() if hasattr(vv, "round") else vv)
        else:
            print(v.round(4).to_string() if hasattr(v, "round") else v)


def site_table():
    """Per-site patient AUROC (mean over cohorts) and seven-day WAPE of facility forecasts."""
    rows = []
    evs = {tag: facility_run.run(tag, "cohere", 0, verbose=False)[1] for tag in TAGS}
    for s, name in enumerate(SITES):
        r = dict(site=name)
        for key in ["cohere", "lgbm"]:
            r[f"auc_{key}"] = np.mean([site_aurocs(load_run(tag, key), prep(tag)["lm"])[s] for tag in TAGS])
        r["n_test"] = int(np.mean([(load_run(tag, "cohere")["split"]["te"] & (prep(tag)["lm"].site.values == s)).sum() for tag in TAGS]))
        for key in ["td", "bu", "mint"]:
            w = []
            for tag in TAGS:
                te = evs[tag]["te"]
                y = np.array([x["y"][s].sum() for x in te]); f = np.array([x[key][s].sum() for x in te])
                w.append(100 * np.abs(y - f).sum() / y.sum())
            r[f"wape_{key}"] = np.mean(w)
        rows.append(r)
    return pd.DataFrame(rows)
