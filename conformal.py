"""Split conformal prediction: unweighted, covariate-shift weighted, and adaptive (ACI)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def weighted_quantile(scores, weights, w_test, level):
    """Quantile of the weighted empirical distribution of calibration scores with the test
    point's weight placed at +infinity (Tibshirani et al. construction)."""
    o = np.argsort(scores)
    s, w = scores[o], weights[o]
    tot = w.sum() + w_test
    cw = np.cumsum(w) / tot
    k = np.searchsorted(cw, level)
    return s[k] if k < len(s) else np.inf


def shift_weights(Zc, Zt, C=0.3, clip=(0.05, 20)):
    """Density-ratio weights p_test(z)/p_cal(z) from a probabilistic domain classifier."""
    X = np.concatenate([Zc, Zt]); y = np.r_[np.zeros(len(Zc)), np.ones(len(Zt))]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, y)
    p = lambda Z: np.clip(clf.predict_proba((Z - mu) / sd)[:, 1], 0.02, 0.98)
    ratio = len(Zc) / len(Zt)
    wc = p(Zc) / (1 - p(Zc)) * ratio
    wt = p(Zt) / (1 - p(Zt)) * ratio
    return np.clip(wc, *clip), np.clip(wt, *clip)


def interval_coverage(y, f, scores_cal, alpha=0.1, wc=None, wt=None):
    n = len(scores_cal)
    if wc is None:
        q = np.quantile(scores_cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        lo, hi = f - q, f + q
    else:
        qs = np.array([weighted_quantile(scores_cal, wc, w, 1 - alpha) for w in wt])
        lo, hi = f - qs, f + qs
    cov = ((y >= lo) & (y <= hi)).mean()
    return cov, np.mean(hi - lo), lo, hi


def aci(y, f, scores_cal, alpha=0.1, gamma=0.01):
    """Adaptive conformal inference: online update of the miscoverage level."""
    hist = list(scores_cal)
    a = alpha; covs, widths = [], []
    for t in range(len(y)):
        lvl = np.clip(1 - a, 0, 1)
        q = np.quantile(hist, lvl) if lvl < 1 else np.max(hist) * 2
        err = float(abs(y[t] - f[t]) > q)
        covs.append(1 - err); widths.append(2 * q)
        a = a + gamma * (alpha - err)
        hist.append(abs(y[t] - f[t]))
    return np.mean(covs), np.mean(widths)


def mondrian_sets(p_cal, y_cal, p_te, alpha=0.1, wc=None, wt=None):
    """Class-conditional conformal prediction sets for a binary outcome.
    Score for label c: 1 - p(c). Returns boolean set membership (n_te, 2)."""
    sets = np.zeros((len(p_te), 2), bool)
    for c in (0, 1):
        pc = p_cal if c == 1 else 1 - p_cal
        sc = 1 - pc[y_cal == c]
        st = 1 - (p_te if c == 1 else 1 - p_te)
        if wc is None:
            n = len(sc)
            q = np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
            sets[:, c] = st <= q
        else:
            w = wc[y_cal == c]
            qs = np.array([weighted_quantile(sc, w, x, 1 - alpha) for x in wt])
            sets[:, c] = st <= qs
    return sets


def set_metrics(sets, y):
    cov = sets[np.arange(len(y)), y.astype(int)].mean()
    cov1 = sets[y == 1, 1].mean(); cov0 = sets[y == 0, 0].mean()
    size = sets.sum(1).mean(); single = (sets.sum(1) == 1).mean()
    return dict(coverage=cov, cov_pos=cov1, cov_neg=cov0, avg_size=size, singleton=single)
