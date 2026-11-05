"""Training loops: centralised, federated (FedAvg / FedProx with secure aggregation), coherence."""
import copy, time
import numpy as np
import torch
from models import CohereNet, GRUD, survival_nll, to_grid
from dataset import pressure_tokens

torch.set_num_threads(1)


def tensors(D, idx, P):
    lm = D["lm"].iloc[idx]
    out = dict(X=torch.from_numpy(D["X"][idx].astype(np.float32)), T=torch.from_numpy(D["T"][idx]),
               S=torch.from_numpy(D["S"][idx].astype(np.int64)), W=torch.from_numpy(D["W"][idx]),
               PAD=torch.from_numpy(D["PAD"][idx]), static=torch.from_numpy(D["static"][idx]),
               P=torch.from_numpy(pressure_tokens(P, lm.site.values, lm.day.values).astype(np.float32)),
               ebin=torch.from_numpy(lm.ebin.values.astype(np.int64)),
               cens=torch.from_numpy(lm.cens_days.values.astype(np.float32)))
    return out


def _fwd(model, b):
    if isinstance(model, GRUD):
        return model(b["G"], b["M"], b["Dl"], b["static"], b["P"])
    return model(b["X"], b["T"], b["S"], b["W"], b["PAD"], b["static"], b["P"])


def add_grid(t):
    G, M, Dl = to_grid(t["X"].numpy(), t["T"].numpy(), t["PAD"].numpy())
    t["G"], t["M"], t["Dl"] = torch.from_numpy(np.nan_to_num(G)), torch.from_numpy(M), torch.from_numpy(Dl)
    return t


def batches(n, bs, rng):
    p = rng.permutation(n)
    for i in range(0, n, bs):
        yield p[i:i + bs]


def _sub(t, ix):
    return {k: v[ix] for k, v in t.items()}


def predict(model, t, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(t["ebin"]), bs):
            b = {k: v[i:i + bs] for k, v in t.items()}
            out.append(torch.sigmoid(_fwd(model, b)))
    return torch.cat(out).numpy()


def coherence_loss(model, t_coh, groups, gsel):
    """Poisson likelihood of facility-observed next-day oxygen starts given summed day-1 hazards."""
    ix = np.concatenate([groups[g] for g in gsel])
    gid = np.concatenate([np.full(len(groups[g]), k) for k, g in enumerate(gsel)])
    b = _sub(t_coh, torch.from_numpy(ix))
    h1 = torch.sigmoid(_fwd(model, b))[:, 0]
    lam = torch.zeros(len(gsel)).index_add(0, torch.from_numpy(gid), h1).clamp_min(1e-4)
    y = torch.zeros(len(gsel)).index_add(0, torch.from_numpy(gid), (b["ebin"] == 1).float())
    return (lam - y * torch.log(lam)).mean() / 5.0


def train_model(model, t_tr, t_va, epochs=12, lr=2e-3, bs=128, seed=0, patience=3, t_coh=None,
                coh_groups=None, lam_coh=0.0, prox_ref=None, mu=0.0, verbose=False, max_steps=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = np.inf, None, 0
    gkeys = list(coh_groups.keys()) if coh_groups else []
    steps = 0
    for ep in range(epochs):
        model.train()
        for ix in batches(len(t_tr["ebin"]), bs, rng):
            b = _sub(t_tr, torch.from_numpy(ix))
            loss = survival_nll(_fwd(model, b), b["ebin"], b["cens"])
            if lam_coh > 0 and gkeys:
                gsel = [gkeys[i] for i in rng.choice(len(gkeys), 12, replace=False)]
                loss = loss + lam_coh * coherence_loss(model, t_coh, coh_groups, gsel)
            if mu > 0 and prox_ref is not None:
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), prox_ref))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            if max_steps and steps >= max_steps:
                return model
        if t_va is None:
            continue
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([survival_nll(_fwd(model, _sub(t_va, torch.arange(i, min(i + 1024, len(t_va["ebin"]))))),
                                             t_va["ebin"][i:i + 1024], t_va["cens"][i:i + 1024]).item()
                                for i in range(0, len(t_va["ebin"]), 1024)]))
        if verbose:
            print(f"  ep{ep} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------- federated
def secure_sum(vectors, seed=0):
    """Pairwise-mask secure aggregation: each pair (a<b) shares a seed; a adds the mask, b
    subtracts it.  The server only sees masked vectors, whose sum equals the true sum."""
    n = len(vectors)
    masked = [v.clone() for v in vectors]
    for a in range(n):
        for b in range(a + 1, n):
            g = torch.Generator().manual_seed(seed * 1000003 + a * 1009 + b)
            m = torch.randn(vectors[a].shape, generator=g) * 10.0
            masked[a] += m; masked[b] -= m
    return torch.stack(masked).sum(0), masked


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.state_dict().values()])


def unflat(model, vec):
    sd, i = model.state_dict(), 0
    for k, v in sd.items():
        n = v.numel(); sd[k] = vec[i:i + n].view_as(v).to(v.dtype); i += n
    model.load_state_dict(sd)


def fed_train(make_model, site_data, t_va, rounds=10, local_epochs=1, lr=2e-3, seed=0, mu=0.0,
              lam_coh=0.0, site_coh=None, verbose=False):
    torch.manual_seed(seed)
    glob = make_model()
    sizes = np.array([len(t["ebin"]) for t in site_data], float)
    best, best_vec = np.inf, None
    log = []
    for r in range(rounds):
        vecs = []
        for s, t in enumerate(site_data):
            loc = copy.deepcopy(glob)
            ref = [p.detach().clone() for p in glob.parameters()] if mu > 0 else None
            tc, gc = (site_coh[s] if site_coh else (None, None))
            train_model(loc, t, None, epochs=local_epochs, lr=lr, seed=seed * 100 + r * 10 + s,
                        prox_ref=ref, mu=mu, t_coh=tc, coh_groups=gc, lam_coh=lam_coh)
            vecs.append(flat(loc) * sizes[s] / sizes.sum())
        agg, _ = secure_sum(vecs, seed=r)
        unflat(glob, agg)
        glob.eval()
        with torch.no_grad():
            vl = float(np.mean([survival_nll(_fwd(glob, _sub(t_va, torch.arange(i, min(i + 1024, len(t_va["ebin"]))))),
                                             t_va["ebin"][i:i + 1024], t_va["cens"][i:i + 1024]).item()
                                for i in range(0, len(t_va["ebin"]), 1024)]))
        log.append(vl)
        if verbose:
            print(f"  round {r} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_vec = vl, flat(glob).clone()
    unflat(glob, best_vec)
    return glob, log
