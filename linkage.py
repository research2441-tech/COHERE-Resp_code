"""Privacy-preserving record linkage with calibrated, null-normalised match probabilities.

Quasi-identifiers are encoded field-wise as Bloom filters of character bigrams (no clear-text
comparison).  A logistic model on field Dice similarities gives calibrated pairwise odds, and
each fragment's odds are normalised over its candidate admissions together with a "no admitted
owner" option (orphan community records), giving soft membership probabilities pi_ij.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

M_BITS, K_HASH = 512, 15
FIELDS = ["first", "sur", "dob", "post", "phone"]
SRC_LIST = ["edlab", "lab", "img", "test", "clinic"]
POP = np.array([bin(i).count("1") for i in range(256)], np.uint16)


def _bloom(s, secret="site-key"):
    bits = np.zeros(M_BITS, np.uint8)
    if not s:
        return np.packbits(bits)
    s = f"_{s.lower()}_"
    for a, b in zip(s, s[1:]):
        g = (secret + a + b).encode()
        h1 = int.from_bytes(hashlib.md5(g).digest()[:8], "little")
        h2 = int.from_bytes(hashlib.sha1(g).digest()[:8], "little")
        for k in range(K_HASH):
            bits[(h1 + k * h2) % M_BITS] = 1
    return np.packbits(bits)


def encode(frags):
    """Field-wise Bloom filters, shape (n, 5, 64) uint8, plus missingness flags."""
    cache = {}
    out = np.zeros((len(frags), len(FIELDS), M_BITS // 8), np.uint8)
    miss = np.zeros((len(frags), len(FIELDS)), bool)
    for j, fld in enumerate(FIELDS):
        vals = frags[fld].fillna("").astype(str).values
        for i, v in enumerate(vals):
            if v not in cache:
                cache[v] = _bloom(v)
            out[i, j] = cache[v]
            miss[i, j] = (v == "")
    return out, miss


def dice(a, b):
    inter = POP[np.bitwise_and(a, b)].sum(-1)
    tot = POP[a].sum(-1) + POP[b].sum(-1)
    return np.where(tot > 0, 2.0 * inter / np.maximum(tot, 1), 0.0)


def candidate_pairs(frags, window=(-48, 72)):
    """Fragments (non-ward) x admissions (ward anchors) at the same site within a time window."""
    anc = frags[frags.src == "ward"]
    oth = frags[frags.src != "ward"]
    rows = []
    for s, a in anc.groupby("site"):
        o = oth[oth.site == s]
        at = a.t.values; order = np.argsort(at); at = at[order]; aidx = a.index.values[order]
        lo = np.searchsorted(at, o.t.values + window[0])
        hi = np.searchsorted(at, o.t.values + window[1], side="right")
        for k, (l, h) in enumerate(zip(lo, hi)):
            if h > l:
                rows.append(np.stack([np.full(h - l, o.index.values[k]), aidx[l:h]], 1))
    return np.concatenate(rows, 0)


def pair_features(frags, pairs, bf, miss):
    j, i = pairs[:, 0], pairs[:, 1]
    d = np.stack([dice(bf[j, f], bf[i, f]) for f in range(len(FIELDS))], 1)
    m = (miss[j] | miss[i]).astype(float)
    d = np.where(m > 0, 0.0, d)
    sex = (frags.sex.values[j] == frags.sex.values[i]).astype(float)[:, None]
    src = frags.src.values[j]
    oh = np.stack([(src == s).astype(float) for s in SRC_LIST], 1)
    dt = ((frags.t.values[j] - frags.t.values[i]) / 24.0)[:, None]
    X = np.concatenate([d, d ** 2, m, sex, oh, oh * dt, oh * dt ** 2], 1)
    return X


class SoftLinker:
    def __init__(self):
        self.clf = LogisticRegression(C=1.0, max_iter=2000)
        self.kappa = 1.0

    def fit(self, frags, pairs, X, train_mask):
        y = (frags.pid.values[pairs[:, 0]] == frags.pid.values[pairs[:, 1]]) & (frags.pid.values[pairs[:, 0]] >= 0)
        self.clf.fit(X[train_mask], y[train_mask])
        odds = self._odds(X)
        # fit kappa (prior odds that a fragment has no admitted owner) by maximum likelihood
        best, bk = -np.inf, 1.0
        for k in np.exp(np.linspace(-4, 4, 41)):
            ll = self._assign_ll(frags, pairs[train_mask], odds[train_mask], y[train_mask], k)
            if ll > best:
                best, bk = ll, k
        self.kappa = bk
        return self

    def _odds(self, X):
        p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return p / (1 - p)

    @staticmethod
    def _normalise(pairs, odds, kappa):
        j = pairs[:, 0]
        uj, inv = np.unique(j, return_inverse=True)
        tot = np.bincount(inv, weights=odds)
        return odds / (kappa + tot[inv])

    def _assign_ll(self, frags, pairs, odds, y, kappa):
        pi = self._normalise(pairs, odds, kappa)
        j = pairs[:, 0]
        uj, inv = np.unique(j, return_inverse=True)
        tot = np.bincount(inv, weights=pi)
        has = np.bincount(inv, weights=y.astype(float)) > 0
        p_true = np.bincount(inv, weights=pi * y)
        lik = np.where(has, p_true, 1 - tot)
        return np.log(np.clip(lik, 1e-9, 1)).sum()

    def predict(self, pairs, X):
        return self._normalise(pairs, self._odds(X), self.kappa)


def deterministic_links(frags, pairs):
    """Exact agreement on surname, date of birth and sex (classical deterministic rule)."""
    j, i = pairs[:, 0], pairs[:, 1]
    f = frags
    ok = ((f.sur.str.lower().values[j] == f.sur.str.lower().values[i]) & (f.dob.values[j] == f.dob.values[i])
          & (f.sex.values[j] == f.sex.values[i]) & (f.sur.values[j] != "") & (f.dob.values[j] != ""))
    w = ok.astype(float)
    # keep nearest-in-time anchor when several agree
    dt = np.abs(f.t.values[j] - f.t.values[i]).astype(float)
    score = np.where(ok, -dt, -np.inf)
    dfp = pd.DataFrame({"j": j, "s": score, "k": np.arange(len(j))})
    best = dfp.loc[dfp.groupby("j").s.idxmax()]
    keep = np.zeros(len(j), bool); keep[best.k.values] = True
    return w * keep


def fellegi_sunter(frags, pairs, X, n_iter=50, thr=0.8):
    """Unsupervised Fellegi-Sunter EM on binary field agreement patterns."""
    G = (X[:, :5] >= thr).astype(float)
    G = np.concatenate([G, X[:, 15:16]], 1)          # + sex agreement
    pm = 0.05
    m = np.full(G.shape[1], 0.9); u = np.full(G.shape[1], 0.1)
    for _ in range(n_iter):
        lm = (G * np.log(m) + (1 - G) * np.log(1 - m)).sum(1) + np.log(pm)
        lu = (G * np.log(u) + (1 - G) * np.log(1 - u)).sum(1) + np.log(1 - pm)
        g = 1 / (1 + np.exp(lu - lm))
        pm = g.mean()
        m = np.clip((g[:, None] * G).sum(0) / g.sum(), 1e-4, 1 - 1e-4)
        u = np.clip(((1 - g)[:, None] * G).sum(0) / (1 - g).sum(), 1e-4, 1 - 1e-4)
    return g


def hard_assign(pairs, w, thr=0.5):
    """One owner per fragment: argmax weight if above threshold."""
    dfp = pd.DataFrame({"j": pairs[:, 0], "w": w, "k": np.arange(len(w))})
    best = dfp.loc[dfp.groupby("j").w.idxmax()]
    best = best[best.w > thr]
    out = np.zeros(len(w)); out[best.k.values] = 1.0
    return out


def linkage_f1(frags, pairs, assign):
    """Precision/recall/F1 over fragments that truly belong to an admitted patient."""
    j, i = pairs[:, 0], pairs[:, 1]
    true = (frags.pid.values[j] == frags.pid.values[i]) & (frags.pid.values[j] >= 0)
    tp = (assign * true).sum(); fp = (assign * (~true)).sum()
    n_true = len(np.unique(j[true]))
    prec = tp / max(tp + fp, 1); rec = tp / max(n_true, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9))


def run_linkage(C, train_end_h, verbose=True):
    frags = C["frags"].reset_index(drop=True)
    bf, miss = encode(frags)
    pairs = candidate_pairs(frags)
    X = pair_features(frags, pairs, bf, miss)
    anchor_t = frags.t.values[pairs[:, 1]]
    tr = anchor_t < train_end_h
    lk = SoftLinker().fit(frags, pairs, X, tr)
    pi = lk.predict(pairs, X)
    det = deterministic_links(frags, pairs)
    fs = fellegi_sunter(frags, pairs, X)
    te = ~tr
    res = {}
    for name, w in [("deterministic", det), ("fellegi_sunter", hard_assign(pairs, fs)),
                    ("soft_argmax", hard_assign(pairs, pi))]:
        r = linkage_f1(frags.iloc[:], pairs[te], w[te])
        res[name] = r
    # calibration of pi on test pairs
    y = (frags.pid.values[pairs[:, 0]] == frags.pid.values[pairs[:, 1]]) & (frags.pid.values[pairs[:, 0]] >= 0)
    res["pi_brier"] = float(np.mean((pi[te] - y[te]) ** 2))
    bins = np.clip((pi[te] * 10).astype(int), 0, 9)
    ece = sum(abs(pi[te][bins == b].mean() - y[te][bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    res["pi_ece"] = float(ece)
    res["kappa"] = float(lk.kappa)
    if verbose:
        print({k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in res.items()})
    return dict(frags=frags, pairs=pairs, pi=pi, det=det, fs=fs, y=y, metrics=res, linker=lk)


if __name__ == "__main__":
    import time
    import simulate
    from config import TRAIN_END, START
    C = simulate.simulate(0)
    te_h = (pd.Timestamp(TRAIN_END) - pd.Timestamp(START)).days * 24
    t = time.time()
    L = run_linkage(C, te_h)
    print(len(L["pairs"]), "pairs", f"{time.time()-t:.0f}s")
