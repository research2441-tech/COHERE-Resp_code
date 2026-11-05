"""Tier C: real ICU time series (PhysioNet/CinC 2019, hospitals A and B).

Each real stay is split into three source systems - bedside monitor (anchor: HR, SpO2, RR,
MAP), nursing chart (temperature, SBP, DBP) and registration (age, sex) - each carrying its own
synthetic quasi-identifiers corrupted at 20%.  Orphan nursing/registration records copied from
other real stays act as distractors.  Task: sepsis onset (Sepsis-3 clinical time, i.e. the
challenge label shifted by +6 h) within 24 h of a landmark at 12/24/36/48 h of ICU stay.
"""
import pickle, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
import simulate, linkage
from models import CohereNet, GRUD, summary_features, to_grid
from train import train_model, predict, fed_train, add_grid
from config import WORK

VIT = ["HR", "O2Sat", "Resp", "MAP", "Temp", "SBP", "DBP"]
F = 9
LM = (12, 24, 36, 48)
LB = 24
MAXT = 50
NB = 4            # 6-hour hazard bins


def load():
    d = pd.read_parquet("sepsis.parquet")
    d["hosp"] = (d.patient_id.str[:2] == "p1").astype(int)
    d["onset"] = np.where(d.has_sepsis, d.sepsis_onset_hour + 6, np.nan)
    return d.reset_index(drop=True)


def fragment(d, seed=0, rate=0.20, orphan_ratio=0.6):
    rng = np.random.default_rng(seed)
    idf = simulate.IdentityFactory(rng)
    adm = np.zeros(len(d), int)
    frags = []
    fid = 0
    ref = pd.Timestamp("2018-01-01")
    for i, r in enumerate(d.itertuples(index=False)):
        adm[i] = int(rng.integers(0, 730 * 24))
        ident = idf.new(r.age, r.gender, ref + pd.Timedelta(hours=int(adm[i])))
        for src, dt, rr in [("ward", 0, rate / 3), ("lab", int(rng.integers(0, 4)), rate), ("clinic", -int(rng.integers(0, 12)), rate)]:
            simulate._new_frag(frags, fid, r.hosp, i, src, adm[i] + dt, ident, rng, rr)
            frags[-1]["kind"] = src; frags[-1]["donor"] = i
            fid += 1
    n_orph = int(orphan_ratio * len(d))
    for _ in range(n_orph):
        donor = int(rng.integers(0, len(d)))
        h = int(d.hosp.values[donor])
        ident = idf.new(np.clip(rng.normal(60, 18), 18, 95), rng.integers(0, 2), ref)
        t = int(rng.integers(0, 730 * 24))
        src = "lab" if rng.random() < 0.5 else "clinic"
        simulate._new_frag(frags, fid, h, -1, src, t, ident, rng, rate)
        frags[-1]["kind"] = src; frags[-1]["donor"] = donor
        fid += 1
    return adm, pd.DataFrame(frags)


def landmarks(d):
    rows = []
    for i, r in enumerate(d.itertuples(index=False)):
        for tau in LM:
            if r.icu_hours <= tau or (np.isfinite(r.onset) and r.onset <= tau):
                continue
            if np.isfinite(r.onset) and r.onset <= tau + NB * 6:
                eb = int(np.ceil((r.onset - tau) / 6.0))
            else:
                eb = 0
            cens = (r.icu_hours - tau) / 6.0 if eb == 0 else NB + 1
            rows.append((i, int(r.hosp), tau, eb, min(cens, NB + 1), int(eb > 0)))
    return pd.DataFrame(rows, columns=["pid", "site", "tau", "ebin", "cens_days", "y"])


def build(d, frags, links, lm, stats=None):
    """links: dict pid -> list of (fid, weight)."""
    N = len(lm)
    X = np.full((N, MAXT, F), np.nan, np.float32); T = np.zeros((N, MAXT), np.float32)
    S = np.zeros((N, MAXT), np.int64); W = np.zeros((N, MAXT), np.float32); PAD = np.ones((N, MAXT), bool)
    vit = {v: d[v].values for v in VIT}
    fk = frags.set_index("fid")
    for n, r in enumerate(lm.itertuples(index=False)):
        hrs = np.arange(max(0, r.tau - LB), r.tau)
        rows = []
        mon = np.stack([vit[v][r.pid][hrs] for v in VIT[:4]], 1)
        for k, h in enumerate(hrs):
            x = np.full(F, np.nan); x[:4] = mon[k]
            if np.isfinite(x[:4]).any():
                rows.append((x, (h - r.tau) / 24.0, 0, 1.0))
        for fid, w in links.get(r.pid, []):
            kind, donor = fk.at[fid, "kind"], int(fk.at[fid, "donor"])
            if kind == "lab":
                nv = np.stack([vit[v][donor] for v in VIT[4:]], 1)
                for h in hrs:
                    if h < len(nv) and np.isfinite(nv[h]).any():
                        x = np.full(F, np.nan); x[4:7] = nv[h]
                        rows.append((x, (h - r.tau) / 24.0, 2, w))
            else:
                x = np.full(F, np.nan); x[7] = (d.age.values[donor] - 60) / 16; x[8] = d.gender.values[donor]
                rows.append((x, -LB / 24.0 - 0.05, 5, w))
        rows = rows[-MAXT:] if len(rows) > MAXT else rows
        for k, (x, t, s, w) in enumerate(rows):
            X[n, k] = x; T[n, k] = t; S[n, k] = s; W[n, k] = w; PAD[n, k] = False
    if stats is None:
        mu = np.nanmean(X.reshape(-1, F), 0); sd = np.nanstd(X.reshape(-1, F), 0) + 1e-6
        mu[7:] = 0; sd[7:] = 1
        stats = (mu, sd)
    X = (X - stats[0]) / stats[1]
    static = (lm.tau.values[:, None] / 48.0).astype(np.float32)
    return dict(X=X, T=T, S=S, W=W, PAD=PAD, static=static, stats=stats)


def to_t(B, ix, lm):
    P = torch.zeros(len(ix), 21, 16)
    return dict(X=torch.from_numpy(B["X"][ix]), T=torch.from_numpy(B["T"][ix]), S=torch.from_numpy(B["S"][ix]),
                W=torch.from_numpy(B["W"][ix]), PAD=torch.from_numpy(B["PAD"][ix]), static=torch.from_numpy(B["static"][ix]),
                P=P, ebin=torch.from_numpy(lm.ebin.values[ix].astype(np.int64)),
                cens=torch.from_numpy(lm.cens_days.values[ix].astype(np.float32)))


def prepare(seed=0):
    t0 = time.time()
    d = load()
    adm, frags = fragment(d, seed)
    frags["t"] = frags["t"].astype(float)
    bf, miss = linkage.encode(frags)
    pairs = linkage.candidate_pairs(frags, window=(-24, 36))
    Xp = linkage.pair_features(frags, pairs, bf, miss)
    rng = np.random.default_rng(seed)
    pid_split = rng.random(len(d))          # 0-0.6 train, 0.6-0.8 cal, 0.8-1 test
    anc_p = frags.pid.values[pairs[:, 1]]
    trm = pid_split[anc_p] < 0.6
    lk = linkage.SoftLinker().fit(frags, pairs, Xp, trm)
    pi = lk.predict(pairs, Xp)
    hard = linkage.hard_assign(pairs, pi)
    det = linkage.deterministic_links(frags, pairs)
    te = ~trm
    lmet = {k: linkage.linkage_f1(frags, pairs[te], w[te]) for k, w in [("soft_argmax", hard), ("deterministic", det)]}
    j, i = pairs[:, 0], pairs[:, 1]
    lm = landmarks(d)
    out = dict(lm=lm, split=pid_split[lm.pid.values], link=lmet, data={})
    for mode in ["soft", "hard", "det", "oracle"]:
        links = {}
        if mode == "oracle":
            own = frags[(frags.src != "ward") & (frags.pid >= 0)]
            for f_, p_ in zip(own.fid.values, own.pid.values):
                links.setdefault(p_, []).append((f_, 1.0))
        else:
            w = {"soft": pi, "hard": hard, "det": det}[mode]
            keep = (w > 0.02) & (frags.src.values[j] != "ward")
            for a, f_, ww in zip(frags.pid.values[i[keep]], frags.fid.values[j[keep]], w[keep]):
                links.setdefault(a, []).append((f_, float(ww)))
        st = out["data"]["soft"]["stats"] if out["data"] else None
        out["data"][mode] = build(d, frags, links, lm, st)
    pickle.dump(out, open(f"{WORK}/tierC_prep.pkl", "wb"), protocol=4)
    print("tierC prepared", len(lm), "pos", lm.y.mean().round(4), {k: round(v["f1"], 4) for k, v in lmet.items()},
          f"{time.time()-t0:.0f}s", flush=True)
    return out


def evaluate(y, r):
    return dict(AUROC=roc_auc_score(y, r), AUPRC=average_precision_score(y, r))


def run_all(seeds=(0, 1, 2), epochs=6, out=None):
    R = pickle.load(open(f"{WORK}/tierC_prep.pkl", "rb"))
    lm, sp = R["lm"], R["split"]
    tr, ca, te = sp < 0.6, (sp >= 0.6) & (sp < 0.8), sp >= 0.8
    y = lm.y.values
    res = []
    preds = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        va = tr & (rng.random(len(lm)) < 0.12); trn = tr & ~va
        for name, mode, use_pi in [("COHERE (soft PMA)", "soft", True), ("Transformer (hard)", "hard", False),
                                   ("Transformer (deterministic)", "det", False), ("Oracle linkage", "oracle", False)]:
            B = R["data"][mode]
            m = CohereNet(F=F, horizon=NB, use_pi=use_pi, pressure="none", n_static=1)
            t0 = time.time()
            train_model(m, to_t(B, np.where(trn)[0], lm), to_t(B, np.where(va)[0], lm), epochs=epochs, seed=seed, bs=256)
            hz = predict(m, to_t(B, np.arange(len(lm)), lm))
            r = 1 - np.prod(1 - hz, 1)
            preds[(name, seed)] = r
            for split_name, msk in [("pooled", te), ("A", te & (lm.site.values == 0)), ("B", te & (lm.site.values == 1))]:
                res.append(dict(model=name, seed=seed, split=split_name, **evaluate(y[msk], r[msk]), train_s=time.time() - t0))
            print(name, seed, res[-3]["AUROC"], flush=True)
        # GRU-D and tabular baselines on hard links
        B = R["data"]["hard"]
        g = GRUD(F=F, horizon=NB, n_static=1)
        ttr = add_grid(to_t(B, np.where(trn)[0], lm)); tva = add_grid(to_t(B, np.where(va)[0], lm))
        train_model(g, ttr, tva, epochs=epochs, seed=seed, bs=256, lr=3e-3)
        hz = np.concatenate([predict(g, add_grid(to_t(B, np.arange(k, min(k + 10000, len(lm))), lm))) for k in range(0, len(lm), 10000)])
        r = 1 - np.prod(1 - hz, 1); preds[("GRU-D (hard)", seed)] = r
        for split_name, msk in [("pooled", te), ("A", te & (lm.site.values == 0)), ("B", te & (lm.site.values == 1))]:
            res.append(dict(model="GRU-D (hard)", seed=seed, split=split_name, **evaluate(y[msk], r[msk])))
        Z = summary_features(B["X"], B["T"], B["PAD"], B["static"])
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        for nm, clf in [("LightGBM (hard)", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                                                colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)),
                        ("Logistic regression (hard)", make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000)))]:
            clf.fit(Z[tr], y[tr]); r = clf.predict_proba(Z)[:, 1]; preds[(nm, seed)] = r
            for split_name, msk in [("pooled", te), ("A", te & (lm.site.values == 0)), ("B", te & (lm.site.values == 1))]:
                res.append(dict(model=nm, seed=seed, split=split_name, **evaluate(y[msk], r[msk])))
        # federated (2 hospitals), local-only and cross-hospital transfer for the soft model
        B = R["data"]["soft"]
        mk = lambda: CohereNet(F=F, horizon=NB, use_pi=True, pressure="none", n_static=1)
        sd = [to_t(B, np.where(trn & (lm.site.values == s))[0], lm) for s in (0, 1)]
        fm, _ = fed_train(mk, sd, to_t(B, np.where(va)[0], lm), rounds=epochs + 2, seed=seed)
        r = 1 - np.prod(1 - predict(fm, to_t(B, np.arange(len(lm)), lm)), 1); preds[("COHERE federated", seed)] = r
        for split_name, msk in [("pooled", te), ("A", te & (lm.site.values == 0)), ("B", te & (lm.site.values == 1))]:
            res.append(dict(model="COHERE federated", seed=seed, split=split_name, **evaluate(y[msk], r[msk])))
        for s_tr, s_te in [(0, 1), (1, 0)]:
            m = mk()
            train_model(m, to_t(B, np.where(trn & (lm.site.values == s_tr))[0], lm),
                        to_t(B, np.where(va & (lm.site.values == s_tr))[0], lm), epochs=epochs, seed=seed, bs=256)
            r = 1 - np.prod(1 - predict(m, to_t(B, np.arange(len(lm)), lm)), 1)
            preds[(f"local_{s_tr}", seed)] = r
            for nm, msk in [(f"local {'AB'[s_tr]} on {'AB'[s_tr]}", te & (lm.site.values == s_tr)),
                            (f"local {'AB'[s_tr]} on {'AB'[s_te]}", te & (lm.site.values == s_te))]:
                res.append(dict(model=nm, seed=seed, split="transfer", **evaluate(y[msk], r[msk])))
    df = pd.DataFrame(res)
    # cross-hospital conformal: calibrate on hospital A, evaluate sets on hospital B
    import conformal
    r = preds[("COHERE (soft PMA)", seeds[0])]
    B = R["data"]["soft"]
    Z = summary_features(B["X"], B["T"], B["PAD"], B["static"])
    Z = np.nan_to_num(Z[:, ::3])
    cal = ca & (lm.site.values == 0); tst = te & (lm.site.values == 1)
    conf = {}
    s_ = conformal.mondrian_sets(r[cal], y[cal], r[tst], 0.1)
    conf["split"] = conformal.set_metrics(s_, y[tst])
    wc, wt = conformal.shift_weights(Z[cal], Z[tst])
    s_ = conformal.mondrian_sets(r[cal], y[cal], r[tst], 0.1, wc, wt)
    conf["weighted"] = conformal.set_metrics(s_, y[tst])
    print(conf)
    pickle.dump(dict(res=df, preds=preds, lm=lm, split=sp, conformal=conf), open(out or f"{WORK}/tierC_results.pkl", "wb"))
    print(df.groupby(["model", "split"])[["AUROC", "AUPRC"]].agg(["mean", "std"]).round(4))
    return df


def merge(seeds=(0, 1, 2)):
    parts = [pickle.load(open(f"{WORK}/tierC_results_s{s}.pkl", "rb")) for s in seeds]
    res = pd.concat([p["res"] for p in parts], ignore_index=True)
    preds = {}
    for p in parts:
        preds.update(p["preds"])
    out = dict(res=res, preds=preds, lm=parts[0]["lm"], split=parts[0]["split"], conformal=parts[0]["conformal"])
    pickle.dump(out, open(f"{WORK}/tierC_results.pkl", "wb"))
    print(res.groupby(["model", "split"])[["AUROC", "AUPRC"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prepare()
    elif sys.argv[1] == "merge":
        merge()
    else:
        s = int(sys.argv[1])
        run_all(seeds=(s,), out=f"{WORK}/tierC_results_s{s}.pkl")
