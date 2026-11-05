"""Train one patient-level model configuration and save landmark predictions + metrics."""
import json, os, pickle, sys, time
import numpy as np, torch
import pressure, dataset
from config import WORK, SITES
from models import CohereNet, GRUD, summary_features
from train import tensors, add_grid, train_model, predict, fed_train

CFG = {
    "cohere":        dict(kind="net", mode="soft", use_pi=True, pressure="xattn"),
    "cohere_concat": dict(kind="net", mode="soft", use_pi=True, pressure="concat"),
    "cohere_nopress":dict(kind="net", mode="soft", use_pi=True, pressure="none"),
    "cohere_hard":   dict(kind="net", mode="hard", use_pi=False, pressure="xattn"),
    "cohere_det":    dict(kind="net", mode="det", use_pi=False, pressure="xattn"),
    "cohere_fs":     dict(kind="net", mode="fs", use_pi=False, pressure="xattn"),
    "oracle":        dict(kind="net", mode="oracle", use_pi=False, pressure="xattn"),
    "tf_hard":       dict(kind="net", mode="hard", use_pi=False, pressure="none"),
    "grud":          dict(kind="grud", mode="hard", pressure=False),
    "grud_ww":       dict(kind="grud", mode="hard", pressure=True),
    "lr":            dict(kind="tab", mode="hard", model="lr", ww=False),
    "lgbm":          dict(kind="tab", mode="hard", model="lgbm", ww=False),
    "lgbm_ww":       dict(kind="tab", mode="hard", model="lgbm", ww=True),
    "lgbm_det":      dict(kind="tab", mode="det", model="lgbm", ww=False),
    "lr_det":        dict(kind="tab", mode="det", model="lr", ww=False),
    "lgbm_soft":     dict(kind="tab", mode="soft", model="lgbm", ww=False, weighted=True),
    "lgbm_soft_ww":  dict(kind="tab", mode="soft", model="lgbm", ww=True, weighted=True),
    "cohere_fed":    dict(kind="fed", mode="soft", use_pi=True, pressure="xattn", mu=0.0),
    "cohere_fedprox":dict(kind="fed", mode="soft", use_pi=True, pressure="xattn", mu=0.01),
    "local":         dict(kind="local", mode="soft", use_pi=True, pressure="xattn"),
    "cohere_delay":  dict(kind="net", mode="soft", use_pi=True, pressure="xattn", delay=60, lam=0.0),
    "cohere_delay_coh": dict(kind="net", mode="soft", use_pi=True, pressure="xattn", delay=60, lam=1.0),
}


def split_idx(lm, seed, delay=0, holdout_sites=None):
    tr, ca, te = dataset.splits(lm)
    if holdout_sites is not None:
        hs = np.isin(lm.site.values, holdout_sites)
        tr = tr & ~hs; ca = ca & ~hs; te = te & hs
    rng = np.random.default_rng(seed)
    up = np.unique(lm.pid.values[tr])
    vp = rng.choice(up, max(len(up) // 10, 1), replace=False)
    va = tr & np.isin(lm.pid.values, vp)
    trn = tr & ~va
    delayed = np.zeros(len(lm), bool)
    if delay:
        cut = lm.lm_abs.values[tr].max() - delay * 24
        delayed = trn & (lm.lm_abs.values > cut)
        trn = trn & ~delayed
    return trn, va, ca, te, delayed


def coh_groups(lm, mask):
    g = {}
    for i in np.where(mask)[0]:
        g.setdefault((lm.site.values[i], lm.day.values[i]), []).append(i)
    return {k: np.array(v) for k, v in g.items() if len(v) >= 2}


def run(tag, name, seed=0, holdout=None, epochs=7, out_tag=None):
    cfg = CFG[name]
    R = pickle.load(open(f"{WORK}/prep_{tag}.pkl", "rb"))
    P = R["pressure"]
    D = pickle.load(open(f"{WORK}/prep_{tag}_{cfg['mode']}.pkl", "rb")); lm = D["lm"]
    del R
    trn, va, ca, te, delayed = split_idx(lm, seed, cfg.get("delay", 0), holdout)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time(); n_par = 0; extra = {}
    allix = np.arange(len(lm))
    if cfg["kind"] in ("net", "fed", "local"):
        mk = lambda: CohereNet(use_pi=cfg["use_pi"], pressure=cfg["pressure"])
        tva = tensors(D, np.where(va)[0], P)
        if cfg["kind"] == "net":
            m = mk(); ttr = tensors(D, np.where(trn)[0], P)
            tc = gc = None
            if cfg.get("lam", 0) > 0:
                dix = np.where(delayed)[0]
                tc = tensors(D, dix, P)
                gl = coh_groups(lm.iloc[dix].reset_index(drop=True), np.ones(len(dix), bool))
                gc = gl
            train_model(m, ttr, tva, epochs=epochs, seed=seed, t_coh=tc, coh_groups=gc, lam_coh=cfg.get("lam", 0))
        elif cfg["kind"] == "fed":
            sd = [tensors(D, np.where(trn & (lm.site.values == s))[0], P) for s in range(len(SITES))
                  if (trn & (lm.site.values == s)).sum() > 0]
            m, log = fed_train(mk, sd, tva, rounds=epochs + 3, seed=seed, mu=cfg["mu"])
            extra["fed_val_curve"] = log
        else:   # local-only: one model per site, each predicts its own site
            hz = np.zeros((len(lm), 7), np.float32)
            for s in range(len(SITES)):
                ms = (lm.site.values == s)
                m = mk()
                ttr = tensors(D, np.where(trn & ms)[0], P); tv = tensors(D, np.where(va & ms)[0], P)
                train_model(m, ttr, tv, epochs=epochs, seed=seed)
                hz[ms] = predict(m, tensors(D, np.where(ms)[0], P))
        if cfg["kind"] != "local":
            n_par = sum(p.numel() for p in m.parameters())
            hz = np.concatenate([predict(m, tensors(D, allix[i:i + 8000], P)) for i in range(0, len(lm), 8000)])
            if cfg["pressure"] == "xattn":
                tt = tensors(D, np.where(te)[0][:4000], P); predict(m, tt)
                extra["gate_mean_test"] = float(m.pca.last_gate.mean())
            t1 = time.time(); predict(m, tensors(D, np.where(te)[0][:2000], P)); extra["ms_per_patient"] = (time.time() - t1) / 2000 * 1000
        risk = 1 - np.prod(1 - hz[:, :3], 1)
    elif cfg["kind"] == "grud":
        m = GRUD(pressure=cfg["pressure"])
        ttr = add_grid(tensors(D, np.where(trn)[0], P)); tva = add_grid(tensors(D, np.where(va)[0], P))
        train_model(m, ttr, tva, epochs=epochs, seed=seed, bs=256, lr=3e-3)
        n_par = sum(p.numel() for p in m.parameters())
        hz = np.concatenate([predict(m, add_grid(tensors(D, allix[i:i + 8000], P))) for i in range(0, len(lm), 8000)])
        risk = 1 - np.prod(1 - hz[:, :3], 1)
    else:
        PT = dataset.pressure_tokens(P, lm.site.values, lm.day.values) if cfg["ww"] else None
        if cfg.get("weighted"):
            from models import weighted_summary
            Zs = []
            for k in range(0, len(lm), 10000):
                sl = slice(k, k + 10000)
                Zs.append(weighted_summary(torch.from_numpy(D["X"][sl].astype(np.float32)), torch.from_numpy(D["T"][sl]),
                                           torch.from_numpy(D["W"][sl]), torch.from_numpy(D["PAD"][sl])).numpy())
            Z = np.concatenate([np.concatenate(Zs), D["static"]] + ([PT[:, -1], PT[:, -7:].mean(1)] if PT is not None else []), 1)
        else:
            Z = summary_features(D["X"], D["T"], D["PAD"], D["static"], PT)
        y = lm.y72.values
        fitm = trn | va
        if cfg["model"] == "lr":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        else:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)
        clf.fit(Z[fitm], y[fitm])
        risk = clf.predict_proba(Z)[:, 1]; hz = None
    extra["train_s"] = time.time() - t0; extra["n_params"] = int(n_par)
    if cfg["kind"] in ("net", "grud", "fed"):
        from models import survival_nll
        hv = torch.from_numpy(np.clip(hz[va], 1e-6, 1 - 1e-6))
        extra["val_nll"] = float(survival_nll(torch.log(hv / (1 - hv)), torch.from_numpy(lm.ebin.values[va].astype(np.int64)),
                                              torch.from_numpy(lm.cens_days.values[va].astype(np.float32))))
    out = dict(name=name, tag=tag, seed=seed, holdout=holdout, risk=risk.astype(np.float32),
               hz=None if hz is None else hz.astype(np.float32), split=dict(trn=trn, va=va, ca=ca, te=te), extra=extra)
    ot = out_tag or f"{tag}__{name}__{seed}" + (f"__ho{'-'.join(map(str, holdout))}" if holdout else "")
    os.makedirs(f"{WORK}/runs", exist_ok=True)
    pickle.dump(out, open(f"{WORK}/runs/{ot}.pkl", "wb"))
    from sklearn.metrics import roc_auc_score
    y = lm.y72.values
    print(ot, "AUROC", round(roc_auc_score(y[te], risk[te]), 4), f"{extra['train_s']:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    tag, name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ho = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    run(tag, name, seed, ho)
