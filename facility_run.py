"""Facility-level evaluation for a given cohort tag and patient-model run."""
import pickle, sys
import numpy as np, pandas as pd
import facility
from config import WORK, START, TRAIN_END, CAL_END

def day_index(s):
    return (pd.Timestamp(s) - pd.Timestamp(START)).days

def run(tag, model, seed=0, verbose=True):
    R = pickle.load(open(f"{WORK}/prep_{tag}.pkl", "rb"))
    run = pickle.load(open(f"{WORK}/runs/{tag}__{model}__{seed}.pkl", "rb"))
    lm = R["lm"]
    ev = facility.evaluate(R["C"], dict(lm=lm, hz=run["hz"]), R["pressure"], day_index(TRAIN_END) + 1,
                           day_index(CAL_END) + 1, R["C"]["facility"]["o2_start"].shape[1])
    rows = []
    for key, name in [("naive", "Seasonal naive (7-day mean)"), ("td_noww", "Direct ridge, no wastewater"),
                      ("td", "Direct ridge + wastewater"), ("td_gbm", "Direct LightGBM + wastewater"),
                      ("bu_noww", "Bottom-up, admissions without wastewater"), ("bu", "Bottom-up (patient hazards)"),
                      ("mint", "COHERE-Resp (MinT reconciled)")]:
        m = facility.metrics(ev["te"], key)
        rows.append(dict(method=name, key=key, MAE=m["MAE"], RMSE=m["RMSE"], MAPE7=m["MAPE7"], WAPE7=m["WAPE7"],
                         **{f"MAE_h{h+1}": m["MAE_h"][h] for h in range(7)}))
    df = pd.DataFrame(rows)
    if verbose:
        print(df[["method", "MAE", "RMSE", "MAPE7", "WAPE7"]].round(3).to_string(index=False))
    return df, ev

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
