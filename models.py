"""COHERE-Resp patient module and neural baselines."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

N_SRC = 6
P_DIM = 16


class TimeEnc(nn.Module):
    def __init__(self, d, n_freq=8):
        super().__init__()
        self.freq = nn.Parameter(torch.exp(torch.linspace(math.log(0.2), math.log(12.0), n_freq)), requires_grad=False)
        self.lin = nn.Linear(2 * n_freq + 1, d)

    def forward(self, t):
        a = t.unsqueeze(-1) * self.freq
        return self.lin(torch.cat([torch.sin(a), torch.cos(a), t.unsqueeze(-1)], -1))


class MembershipAttention(nn.Module):
    """Multi-head self-attention with (i) a learned time-decay penalty and (ii) an additive
    log-membership bias beta_h * log(pi_j), so that tokens from uncertain links are attenuated."""

    def __init__(self, d, heads, use_pi=True):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.log_gamma = nn.Parameter(torch.full((heads,), -1.0))
        self.beta = nn.Parameter(torch.ones(heads))
        self.use_pi = use_pi

    def forward(self, x, t, logw, pad):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        dt = (t.unsqueeze(-1) - t.unsqueeze(-2)).abs().unsqueeze(1)
        logits = logits - Fn.softplus(self.log_gamma).view(1, -1, 1, 1) * dt
        if self.use_pi:
            logits = logits + self.beta.view(1, -1, 1, 1) * logw.view(B, 1, 1, L)
        logits = logits.masked_fill(pad.view(B, 1, 1, L), -1e9)
        a = torch.softmax(logits, -1)
        y = (a @ v).transpose(1, 2).reshape(B, L, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d, heads, use_pi):
        super().__init__()
        self.att = MembershipAttention(d, heads, use_pi)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(0.1)

    def forward(self, x, t, logw, pad):
        x = x + self.drop(self.att(self.n1(x), t, logw, pad))
        return x + self.drop(self.ff(self.n2(x)))


class PressureCrossAttention(nn.Module):
    """Patient summary queries the recent wastewater-pressure history; a gate decides how much
    community context enters the patient representation."""

    def __init__(self, d, window=21, heads=2):
        super().__init__()
        self.inp = nn.Linear(P_DIM, d)
        self.pos = nn.Parameter(torch.zeros(window, d))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.gate = nn.Linear(2 * d, d)
        self.proj = nn.Linear(d, d)

    def forward(self, h, P):
        p = self.inp(P) + self.pos
        c, w = self.att(h.unsqueeze(1), p, p, need_weights=True)
        c = c.squeeze(1)
        g = torch.sigmoid(self.gate(torch.cat([h, c], -1)))
        self.last_gate, self.last_attn = g.detach(), w.detach()
        return h + g * self.proj(c)


def weighted_summary(X, T, W, PAD, use_pi=True, thr=0.5):
    """Membership-weighted evidence summary per variable: weighted mean, min, max and last value
    over confidently linked tokens, recency of the last value, and membership-weighted count."""
    m = torch.isfinite(X) & ~PAD.unsqueeze(-1)
    x = torch.nan_to_num(X, 0.0)
    w = (W if use_pi else (W > 0).float()).unsqueeze(-1) * m.float()
    conf = m & ((W > thr) if use_pi else (W > 0)).unsqueeze(-1)
    cnt = w.sum(1)
    mean = (w * x).sum(1) / cnt.clamp_min(1e-6)
    big = torch.tensor(1e4)
    mx = torch.where(conf, x, -big).amax(1); mx = torch.where(conf.any(1), mx, torch.zeros_like(mx))
    mn = torch.where(conf, x, big).amin(1); mn = torch.where(conf.any(1), mn, torch.zeros_like(mn))
    L = X.shape[1]
    pos = torch.arange(L).view(1, L, 1).expand_as(x)
    lastpos = torch.where(conf, pos, torch.full_like(pos, -1)).amax(1)
    has = lastpos >= 0
    last = torch.gather(x, 1, lastpos.clamp_min(0).unsqueeze(1)).squeeze(1) * has.float()
    tl = torch.gather(T.unsqueeze(-1).expand_as(x), 1, lastpos.clamp_min(0).unsqueeze(1)).squeeze(1)
    rec = torch.where(has, -tl, torch.full_like(tl, 4.0))
    return torch.cat([mean, mn, mx, last, rec, torch.log1p(cnt), has.float()], -1)


class CohereNet(nn.Module):
    def __init__(self, F=19, d=32, heads=2, layers=2, horizon=7, use_pi=True, pressure="xattn",
                 n_static=3):
        super().__init__()
        self.use_pi, self.pressure = use_pi, pressure
        self.val = nn.Linear(2 * F, d)
        self.src = nn.Embedding(N_SRC, d)
        self.tenc = TimeEnc(d)
        self.blocks = nn.ModuleList([Block(d, heads, use_pi) for _ in range(layers)])
        self.pool_q = nn.Parameter(torch.randn(d) * 0.1)
        self.pool_k = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        if pressure == "xattn":
            self.pca = PressureCrossAttention(d)
        elif pressure == "concat":
            self.pcat = nn.Linear(P_DIM * 2, d)
        self.head = nn.Sequential(nn.Linear(d + n_static, d), nn.GELU(), nn.Linear(d, horizon))

    def forward(self, X, T, S, W, PAD, static, P=None):
        m = torch.isfinite(X).float()
        x = torch.nan_to_num(X, 0.0)
        e = self.val(torch.cat([x * m, m], -1)) + self.src(S.long()) + self.tenc(T)
        logw = torch.log(W.clamp_min(1e-4))
        if not self.use_pi:
            logw = torch.zeros_like(logw)
        for b in self.blocks:
            e = b(e, T, logw, PAD)
        e = self.norm(e)
        sc = (self.pool_k(e) @ self.pool_q) / math.sqrt(e.shape[-1]) + (logw if self.use_pi else 0)
        sc = sc.masked_fill(PAD, -1e9)
        h = (torch.softmax(sc, -1).unsqueeze(-1) * e).sum(1)
        if self.pressure == "xattn":
            h = self.pca(h, P)
        elif self.pressure == "concat":
            h = h + self.pcat(torch.cat([P[:, -1], P.mean(1)], -1))
        return self.head(torch.cat([h, static], -1))          # hazard logits per day


class GRUD(nn.Module):
    """GRU-D on a 2-hour grid (decay of inputs toward the empirical mean and of the hidden state)."""

    def __init__(self, F=19, d=48, horizon=7, n_static=3, pressure=False):
        super().__init__()
        self.F = F
        self.gx = nn.Parameter(torch.zeros(F)); self.wx = nn.Parameter(torch.full((F,), 0.1))
        self.wh = nn.Linear(F, d)
        self.cell = nn.GRUCell(3 * F, d)
        self.pressure = pressure
        extra = 2 * P_DIM if pressure else 0
        self.head = nn.Sequential(nn.Linear(d + n_static + extra, d), nn.GELU(), nn.Linear(d, horizon))

    def forward(self, Xg, Mg, Dg, static, P=None):
        B, Tn, F = Xg.shape
        h = Xg.new_zeros(B, self.cell.hidden_size)
        last = Xg.new_zeros(B, F)
        for k in range(Tn):
            x, m, dl = Xg[:, k], Mg[:, k], Dg[:, k]
            gx = torch.exp(-Fn.relu(self.wx * dl + self.gx))
            last = torch.where(m > 0, x, last)
            xh = m * x + (1 - m) * gx * last
            gh = torch.exp(-Fn.relu(self.wh(dl)))
            h = self.cell(torch.cat([xh, m, dl], -1), h * gh)
        z = [h, static]
        if self.pressure:
            z += [P[:, -1], P.mean(1)]
        return self.head(torch.cat(z, -1))


def to_grid(X, T, PAD, bins=36, width=1.0 / 12):
    """Tokens -> regular 2 h grid over the last 72 h (last observation per bin), masks, deltas."""
    N, L, F = X.shape
    G = np.zeros((N, bins, F), np.float32); M = np.zeros((N, bins, F), np.float32)
    idx = np.floor((T + bins * width) / width).astype(int)
    valid = (~PAD) & (idx >= 0) & (idx < bins)
    idx = np.clip(idx, 0, bins - 1)
    Xf = X.astype(np.float32)
    early = (~PAD) & (idx == 0) & (T < -bins * width)        # older evidence (clinic) -> first bin
    valid = valid | early
    for l in range(L):
        v = valid[:, l]
        if not v.any():
            continue
        xs = Xf[v, l]; ok = np.isfinite(xs)
        rows = np.where(v)[0]; b = idx[v, l]
        cur = G[rows, b]; cm = M[rows, b]
        G[rows, b] = np.where(ok, xs, cur); M[rows, b] = np.maximum(cm, ok)
    D = np.zeros_like(M)
    for k in range(1, bins):
        D[:, k] = np.where(M[:, k - 1] > 0, width, D[:, k - 1] + width)
    return G, M, D * 12


def summary_features(X, T, PAD, static, P=None):
    """Hand-crafted summaries for tabular baselines (last, mean, min, max, count, recency)."""
    Xf = X.astype(np.float32).copy()
    Xf[PAD] = np.nan
    with np.errstate(all="ignore"):
        last = np.full(Xf.shape[::2], np.nan, np.float32)
        rec = np.full(last.shape, 9.0, np.float32)
        for l in range(Xf.shape[1]):
            ok = np.isfinite(Xf[:, l])
            last = np.where(ok, Xf[:, l], last)
            rec = np.where(ok, -T[:, l:l + 1], rec)
        mean = np.nanmean(Xf, 1); mn = np.nanmin(Xf, 1); mx = np.nanmax(Xf, 1)
        cnt = np.isfinite(Xf).sum(1).astype(np.float32)
    Z = np.concatenate([last, mean, mn, mx, cnt, rec, static], 1)
    if P is not None:
        Z = np.concatenate([Z, P[:, -1], P[:, -7:].mean(1)], 1)
    return Z


def hazards_to_risk(logits, days=3):
    h = torch.sigmoid(logits)
    return 1 - torch.prod(1 - h[:, :days], 1)


def survival_nll(logits, ebin, cens):
    """Discrete-time survival negative log-likelihood with right censoring."""
    H = logits.shape[1]
    k = torch.arange(1, H + 1, device=logits.device).view(1, -1).float()
    logh = Fn.logsigmoid(logits); log1mh = Fn.logsigmoid(-logits)
    ev = ebin > 0
    surv_upto = torch.where(ev.unsqueeze(1), k < ebin.unsqueeze(1).float(), k <= torch.floor(cens).unsqueeze(1))
    ll = (log1mh * surv_upto.float()).sum(1)
    ll = ll + torch.where(ev, logh.gather(1, (ebin.clamp_min(1) - 1).long().unsqueeze(1)).squeeze(1), torch.zeros_like(ll))
    return -ll.mean()
