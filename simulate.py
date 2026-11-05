"""Ten-site respiratory-admission cohort driven by real wastewater curves.

Admissions at each site follow the site's real sewershed concentration (leading by a site
lag).  Each patient has a latent physiological derangement z(t) whose trajectory depends on
age, comorbidity, vaccination, pathogen, circulating variant and hospital strain.  Crossing
the oxygen threshold is the deterioration event.  Evidence is split across five source
systems, each carrying its own copy of quasi-identifiers corrupted at a controlled rate, and
mixed with orphan community records and household members (hard negatives).
"""
import numpy as np
import pandas as pd
from config import SITES, VARIANT_SEV, WORK
import pressure

H = 240                     # hours simulated per admission
TAU_O2, TAU_ICU, TAU_DEATH = 1.6, 2.4, 3.1
PATHOGENS = ["SARS-CoV-2", "Influenza", "RSV"]
FEATS = ["SpO2", "RR", "HR", "Temp", "SBP",                  # 0-4 vitals (anchor)
         "CRP", "Lymph", "LDH", "Ddimer", "Creat",           # 5-9 labs
         "Img1", "Img2", "Img3", "Img4",                     # 10-13 imaging report embedding
         "Test_SARS2", "Test_Flu", "Test_RSV",               # 14-16 pathogen test result
         "Comorb", "Vacc"]                                   # 17-18 clinic record
F = len(FEATS)
SOURCES = ["ward", "edlab", "lab", "img", "test", "clinic"]


# ----------------------------------------------------------------------------- identities
def _syllable_names(rng, n, lo=2, hi=3):
    syl = ["ka", "ra", "lin", "do", "mi", "sa", "an", "ve", "to", "ri", "el", "on", "ma", "ne",
           "ar", "is", "ul", "be", "da", "jo", "ki", "lo", "pe", "su", "ta", "vi", "ze", "ha",
           "gu", "fi", "wen", "sh", "ch", "ny", "or", "em", "al", "us", "ia", "ot"]
    out = set()
    while len(out) < n:
        k = rng.integers(lo, hi + 1)
        out.add("".join(rng.choice(syl, k)).capitalize())
    return np.array(sorted(out))


class IdentityFactory:
    def __init__(self, rng):
        self.rng = rng
        self.surnames = _syllable_names(rng, 700, 2, 3)
        self.first = _syllable_names(rng, 260, 2, 2)
        zs = 1.0 / np.arange(1, len(self.surnames) + 1) ** 1.05
        zf = 1.0 / np.arange(1, len(self.first) + 1) ** 1.1
        self.ps, self.pf = zs / zs.sum(), zf / zf.sum()
        letters = np.array(list("ABCEGHJKLMNPRSTVXY"))
        self.fsa = np.array([a + str(d) + b for a in letters[:12] for d in range(10) for b in letters[:6]])

    def new(self, age, sex, ref_date, household=None):
        r = self.rng
        if household is None:
            sur = r.choice(self.surnames, p=self.ps)
            post = r.choice(self.fsa) + " " + str(r.integers(1, 10)) + r.choice(list("ABCEGHJKLMN")) + str(r.integers(0, 10))
            phone = f"{r.integers(0, 10000):04d}"
        else:
            sur, post, phone = household["sur"], household["post"], household["phone"]
        first = r.choice(self.first, p=self.pf)
        dob = ref_date - pd.Timedelta(days=int(age * 365.25 + r.integers(0, 365)))
        return {"first": first, "sur": sur, "dob": dob.strftime("%Y%m%d"), "sex": int(sex),
                "post": post, "phone": phone}


def _typo(rng, s):
    if len(s) < 2:
        return s
    i = rng.integers(0, len(s))
    op = rng.integers(0, 3)
    c = rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
    if op == 0:
        return s[:i] + c + s[i + 1:]
    if op == 1:
        return s[:i] + s[i + 1:]
    return s[:i] + c + s[i:]


def corrupt(rng, ident, rate):
    """Field-wise corruption: typos, deletions, transpositions, missingness."""
    out = dict(ident)
    for fld in ["first", "sur", "dob", "post", "phone"]:
        if rng.random() < rate:
            u = rng.random()
            v = out[fld]
            if u < 0.25:
                out[fld] = ""                                   # missing
            elif fld == "dob" and u < 0.55:
                out[fld] = v[:4] + v[6:8] + v[4:6]              # day-month swap
            elif fld == "first" and u < 0.45:
                out[fld] = v[:3]                                 # truncated / nickname
            else:
                out[fld] = _typo(rng, v.lower()).capitalize() if fld in ("first", "sur") else _typo(rng, v)
    if rng.random() < rate / 10:
        out["sex"] = 1 - out["sex"]
    return out


# ----------------------------------------------------------------------------- physiology
def _trajectory(rng, eta, pathogen, z0):
    """Latent derangement: severity-driven drift, AR(1) fluctuation and stochastic complications
    (secondary infection, thrombo-embolic events) whose probability rises with severity."""
    T = {0: 70.0, 1: 30.0, 2: 50.0}[pathogen]
    t = np.arange(H, dtype=float)
    amp = 2.4 / (1 + np.exp(-eta)) - 0.8
    drift = z0 + amp * (1 - np.exp(-t / T)) - 0.35 * t / H
    noise = np.zeros(H)
    e = rng.normal(0, 0.09, H)
    for k in range(1, H):
        noise[k] = 0.97 * noise[k - 1] + e[k]
    z = drift + noise
    if rng.random() < 0.10 / (1 + np.exp(-(eta + 0.5))):
        ts = int(rng.integers(18, 160))
        z[ts:] += rng.uniform(0.8, 1.6)
    return z


def _vitals(rng, z, pathogen, hours):
    zz = z[hours]
    fever = {0: 0.6, 1: 1.0, 2: 0.5}[pathogen]
    v = np.empty((len(hours), 5))
    v[:, 0] = np.clip(97.5 - 4.2 * zz + rng.normal(0, 1.8, len(zz)), 70, 100)
    v[:, 1] = 15 + 4.0 * zz + rng.normal(0, 2.6, len(zz))
    v[:, 2] = 78 + 9 * zz + 6 * fever + rng.normal(0, 8, len(zz))
    v[:, 3] = 36.9 + 0.5 * fever * np.clip(zz, 0, None) + rng.normal(0, 0.35, len(zz))
    v[:, 4] = 126 - 6 * zz + rng.normal(0, 11, len(zz))
    return v


def _labs(rng, z, pathogen, hours, age):
    zz = z[hours]
    n = len(zz)
    v = np.empty((n, 5))
    v[:, 0] = np.exp(2.4 + 0.55 * zz + (0.3 if pathogen == 0 else 0) + rng.normal(0, 0.45, n))
    v[:, 1] = np.clip(1.5 - (0.35 if pathogen == 0 else 0.2) * zz + rng.normal(0, 0.3, n), 0.1, None)
    v[:, 2] = 210 + 55 * zz + rng.normal(0, 40, n)
    v[:, 3] = np.exp(-0.6 + (0.45 if pathogen == 0 else 0.25) * zz + rng.normal(0, 0.5, n))
    v[:, 4] = 75 + 0.4 * (age - 60) + 9 * zz + rng.normal(0, 15, n)
    return v


_IMG_A = np.array([[0.9, 0.1, 0.0], [0.5, -0.6, 0.3], [0.2, 0.3, -0.7], [-0.4, 0.5, 0.5]])


def _imaging(rng, z, pathogen, hours):
    zz = z[hours]
    patt = np.eye(3)[pathogen]
    base = np.outer(zz, _IMG_A[:, 0]) + 0.6 * (patt @ _IMG_A.T)[None, :]
    return base + rng.normal(0, 0.35, base.shape)


# ----------------------------------------------------------------------------- cohort
def simulate(seed=0, corruption=0.20, null_pressure=False, scale=1.0, household_rate=0.30,
             orphan_ratio=2.0):
    rng = np.random.default_rng(seed)
    P = pressure.build()
    days, truth, vp = P["days"], P["truth"], P["variant_prev"]
    D, S = len(days), len(SITES)
    idf = IdentityFactory(rng)
    site_lag = rng.integers(5, 11, S)
    site_scale = rng.uniform(0.7, 1.3, S) * scale
    base = np.array([1.05, 0.55, 0.40])
    alpha = np.array([1.05, 1.15, 1.10])
    cap = np.round(30 * site_scale).astype(int)

    pats, frags, obs = [], [], []
    o2_start = np.zeros((S, D), int); adm_count = np.zeros((S, D), int); census = np.zeros((S, D), int)
    pid = 0
    fid = 0
    for s in range(S):
        dis_hours = []                     # discharge hours (absolute) of current inpatients
        for d in range(D):
            t0 = d * 24
            dis_hours = [h for h in dis_hours if h > t0]
            cen = len(dis_hours)
            census[s, d] = cen
            strain = 0.0 if null_pressure else max(0.0, cen / cap[s] - 0.75)
            for p in range(3):
                dd = d - site_lag[s]
                x = truth[s, max(dd, 0), p]
                lam = base[p] * site_scale[s] * (np.exp(alpha[p] * x) if np.isfinite(x) else 0.25)
                n = rng.poisson(lam)
                for _ in range(n):
                    if p == 0:
                        age = np.clip(rng.normal(66, 17), 18, 100)
                    elif p == 1:
                        age = np.clip(rng.normal(62, 20), 18, 100)
                    else:
                        age = np.clip(rng.normal(73, 13), 18, 100)
                    sex = rng.integers(0, 2)
                    comorb = rng.poisson(0.6 + age / 45)
                    vax = int(rng.random() < (0.78 if p == 0 else 0.45 if p == 1 else 0.0))
                    if p == 0:
                        var = rng.choice(len(VARIANT_SEV), p=vp[d] / vp[d].sum())
                        vsev = 1.0 if null_pressure else VARIANT_SEV[var]
                    else:
                        var, vsev = -1, 1.0
                    peff = 0.0 if null_pressure else {0: 0.0, 1: -0.15, 2: 0.35}[p]
                    eta = (-0.55 + 0.33 * (age - 65) / 10 + 0.22 * comorb - 0.55 * vax + np.log(vsev)
                           + peff + 1.6 * strain + rng.normal(0, 0.75))
                    z0 = 0.45 + 0.30 / (1 + np.exp(-eta)) + rng.normal(0, 0.38) + 0.3 * strain
                    z = _trajectory(rng, eta, p, z0)
                    adm_h = t0 + rng.integers(0, 24)
                    cross = np.where(z >= TAU_O2)[0]
                    ev = int(cross[0]) if len(cross) else -1
                    zmax = z.max()
                    etype = 0 if ev < 0 else (3 if zmax >= TAU_DEATH else 2 if zmax >= TAU_ICU else 1)
                    if ev < 0:
                        los = int(np.clip(36 + rng.gamma(2.0, 30), 24, H - 1))
                    else:
                        los = int(np.clip(ev + 72 + rng.gamma(2.0, 40), ev + 24, 30 * 24))
                    dis_hours.append(adm_h + los)
                    if ev >= 0:
                        dd_ev = (adm_h + ev) // 24
                        if dd_ev < D:
                            o2_start[s, dd_ev] += 1
                    adm_count[s, d] += 1
                    hh = None
                    ident = idf.new(age, sex, days[d])
                    pats.append(dict(pid=pid, site=s, day=d, adm_h=adm_h, pathogen=p, variant=var,
                                     age=age, sex=sex, comorb=comorb, vax=vax, eta=eta, event_h=ev,
                                     etype=etype, los=min(los, H - 1) if ev < 0 else los,
                                     strain=strain, household=0))
                    _emit(rng, pid, s, adm_h, p, age, comorb, vax, z, ev, los, ident, corruption,
                          frags, obs, fid_ref := [fid])
                    fid = fid_ref[0]
                    # household members (hard negatives) and orphan community records
                    if rng.random() < household_rate:
                        hident = idf.new(np.clip(age + rng.normal(0, 15), 1, 100), rng.integers(0, 2),
                                         days[d], household=ident)
                        fid = _emit_orphan(rng, s, adm_h + rng.integers(-96, 96), hident, corruption,
                                           frags, obs, fid)
                    for _ in range(rng.poisson(orphan_ratio)):
                        oid = idf.new(np.clip(rng.normal(50, 20), 1, 100), rng.integers(0, 2), days[d])
                        fid = _emit_orphan(rng, s, adm_h + rng.integers(-72, 72), oid, corruption,
                                           frags, obs, fid)
                    pid += 1
    pats = pd.DataFrame(pats)
    frags = pd.DataFrame(frags)
    obs_arr = np.concatenate([o[1] for o in obs], 0).astype(np.float32)
    obs_meta = np.concatenate([o[0] for o in obs], 0)
    fac = dict(o2_start=o2_start, admissions=adm_count, census=census, capacity=cap, site_lag=site_lag)
    return dict(patients=pats, frags=frags, obs_vals=obs_arr, obs_meta=obs_meta, facility=fac,
                pressure=P, meta=dict(seed=seed, corruption=corruption, null_pressure=null_pressure))


def _add_obs(obs, fid, times_abs, src, cols, vals):
    meta = np.zeros((len(times_abs), 3))
    meta[:, 0] = fid; meta[:, 1] = times_abs; meta[:, 2] = SOURCES.index(src)
    full = np.full((len(times_abs), F), np.nan)
    full[:, cols] = vals
    obs.append((meta, full))


SOURCE_FIELDS = {"ward": ("first", "sur", "dob", "post", "phone"),
                 "edlab": ("first", "sur", "dob"),
                 "lab": ("first", "sur", "dob", "phone"),
                 "img": ("first", "sur", "dob", "post"),
                 "test": ("first", "sur", "dob"),
                 "clinic": ("first", "sur", "dob", "post", "phone")}


def _new_frag(frags, fid, s, pid, src, t_abs, ident, rng, rate):
    rec = corrupt(rng, ident, rate)
    for fld in ("first", "sur", "dob", "post", "phone"):
        if fld not in SOURCE_FIELDS[src]:
            rec[fld] = ""
    rec.update(fid=fid, site=s, pid=pid, src=src, t=t_abs)
    frags.append(rec)


def _emit(rng, pid, s, adm_h, p, age, comorb, vax, z, ev, los, ident, rate, frags, obs, fid_ref):
    fid = fid_ref[0]
    stop = min(los, H - 1)
    # ward vitals: every 4 h, every 1-2 h when deranged (informative sampling)
    hrs, h = [], 0
    while h <= stop:
        hrs.append(h)
        h += int(rng.integers(1, 4)) if z[h] > 1.1 else int(rng.integers(4, 9))
    hrs = np.array(hrs)
    _new_frag(frags, fid, s, pid, "ward", adm_h, ident, rng, rate / 3)
    _add_obs(obs, fid, adm_h + hrs, "ward", list(range(0, 5)), _vitals(rng, z, p, hrs))
    fid += 1
    # emergency/community lab panel before admission (external laboratory system)
    if rng.random() < 0.8:
        th = int(rng.integers(-8, 2))
        _new_frag(frags, fid, s, pid, "edlab", adm_h + th, ident, rng, rate)
        _add_obs(obs, fid, np.array([adm_h + th]), "edlab", list(range(5, 10)),
                 _labs(rng, z, p, np.array([max(th, 0)]), age))
        fid += 1
    # inpatient laboratory information system: daily panels
    lh = np.arange(int(rng.integers(6, 14)), stop + 1, 24)
    if len(lh):
        _new_frag(frags, fid, s, pid, "lab", adm_h + lh[0], ident, rng, rate)
        _add_obs(obs, fid, adm_h + lh, "lab", list(range(5, 10)), _labs(rng, z, p, lh, age))
        fid += 1
    # imaging centre report embedding
    ih = [int(np.clip(rng.integers(-10, 14), -10, stop))]
    if z[min(36, stop)] > 1.2 and stop > 40:
        ih.append(int(rng.integers(36, 44)))
    ih = np.array(ih)
    _new_frag(frags, fid, s, pid, "img", adm_h + ih[0], ident, rng, rate)
    _add_obs(obs, fid, adm_h + ih, "img", list(range(10, 14)), _imaging(rng, z, p, np.clip(ih, 0, None)))
    fid += 1
    # pathogen test result (turnaround 2-48 h; sometimes resulted in the community before admission)
    tt = int(rng.integers(-30, 0)) if rng.random() < 0.25 else int(rng.integers(2, 49))
    _new_frag(frags, fid, s, pid, "test", adm_h + tt, ident, rng, rate)
    _add_obs(obs, fid, np.array([adm_h + tt]), "test", [14, 15, 16], np.eye(3)[p][None, :])
    fid += 1
    # primary-care / clinic record with comorbidity count and vaccination
    if rng.random() < 0.7:
        _new_frag(frags, fid, s, pid, "clinic", adm_h - int(rng.integers(24, 72)), ident, rng, rate)
        _add_obs(obs, fid, np.array([adm_h - 72]), "clinic", [17, 18], np.array([[comorb, vax]], float))
        fid += 1
    fid_ref[0] = fid


def _emit_orphan(rng, s, t_abs, ident, rate, frags, obs, fid):
    """Community record of a person never admitted: a test or lab panel with its own identity."""
    src = "test" if rng.random() < 0.6 else ("edlab" if rng.random() < 0.6 else "clinic")
    _new_frag(frags, fid, s, -1, src, t_abs, ident, rng, rate)
    zf = np.full(H, max(0.0, rng.normal(0.3, 0.4)))
    if src == "test":
        _add_obs(obs, fid, np.array([t_abs]), "test", [14, 15, 16], np.eye(3)[rng.integers(0, 3)][None, :])
    elif src == "edlab":
        _add_obs(obs, fid, np.array([t_abs]), "edlab", list(range(5, 10)),
                 _labs(rng, zf, int(rng.integers(0, 3)), np.array([0]), 50))
    else:
        _add_obs(obs, fid, np.array([t_abs - 48]), "clinic", [17, 18],
                 np.array([[rng.poisson(1.0), int(rng.random() < 0.6)]], float))
    return fid + 1


if __name__ == "__main__":
    import time, pickle
    t = time.time()
    C = simulate(0)
    p = C["patients"]
    print("patients", len(p), "frags", len(C["frags"]), "obs", len(C["obs_vals"]), f"{time.time()-t:.0f}s")
    print(p.groupby("pathogen").size(), p.groupby("site").size().values)
    at12 = p[(p.event_h < 0) | (p.event_h > 12)]
    y = (at12.event_h > 12) & (at12.event_h <= 84)
    print("already O2 at adm (<12h):", ((p.event_h >= 0) & (p.event_h <= 12)).mean().round(3),
          "72h deterioration among at-risk at 12h:", y.mean().round(3))
    print(p.groupby("pathogen").apply(lambda g: ((g.event_h > 12) & (g.event_h <= 84)).mean()).round(3))
    print("mean census", C["facility"]["census"].mean(1).round(1), "cap", C["facility"]["capacity"])
    print("strain>0 share", (p.strain > 0).mean().round(3))
