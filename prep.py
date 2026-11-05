"""Simulate, link, and cache landmark datasets for every linkage policy."""
import pickle, time, sys
import numpy as np, pandas as pd
import simulate, linkage, dataset
from config import WORK, TRAIN_END, START

def prepare(seed=0, corruption=0.20, null_pressure=False, modes=("soft", "hard", "det", "oracle", "fs"), tag=None):
    t = time.time()
    C = simulate.simulate(seed, corruption=corruption, null_pressure=null_pressure)
    tr_h = ((pd.Timestamp(TRAIN_END) - pd.Timestamp(START)).days + 1) * 24
    L = linkage.run_linkage(C, tr_h, verbose=False)
    lm = dataset.landmarks(C["patients"])
    out = dict(C=C, link_metrics=L["metrics"], lm=lm, pressure=C["pressure"])
    tag = tag or f"s{seed}_c{int(corruption*100)}{'_null' if null_pressure else ''}"
    stats = None
    for m in modes:                         # one file per linkage policy keeps memory bounded
        d = dataset.build(C, L, m, lm=lm, stats=stats)
        stats = d["stats"]
        with open(f"{WORK}/prep_{tag}_{m}.pkl", "wb") as f:
            pickle.dump(d, f, protocol=4)
        del d
    del out["C"]["obs_vals"], out["C"]["obs_meta"]
    out["C"]["frags"] = None
    with open(f"{WORK}/prep_{tag}.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(tag, "landmarks", len(lm), "pos72", lm.y72.mean().round(3), {k: round(v['f1'], 4) for k, v in L['metrics'].items() if isinstance(v, dict)}, f"{time.time()-t:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    corr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    null = len(sys.argv) > 3 and sys.argv[3] == "null"
    prepare(seed, corr, null)
