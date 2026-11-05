"""Run a list of (tag, model, seed, holdout) jobs with N parallel workers; skip finished ones."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from config import WORK

def job_list(phase):
    J = []
    if phase == 1:
        for tag in ["s0_c20", "s1_c20", "s2_c20"]:
            for m in ["cohere", "tf_hard", "lgbm_ww", "lgbm", "lr", "grud", "grud_ww", "cohere_fed", "lgbm_soft_ww"]:
                J.append((tag, m, 0, None))
        for m in ["cohere_nopress", "cohere_concat", "cohere_hard", "cohere_det", "cohere_fs", "oracle",
                  "cohere_fedprox", "local", "cohere_delay", "cohere_delay_coh"]:
            J.append(("s0_c20", m, 0, None))
        for sd in [1, 2]:
            for m in ["cohere", "tf_hard", "cohere_nopress", "cohere_concat", "cohere_hard", "cohere_det",
                      "oracle", "cohere_delay", "cohere_delay_coh"]:
                J.append(("s0_c20", m, sd, None))
    if phase == 2:
        for c in ["c0", "c10", "c30", "c40"]:
            for m in ["cohere", "cohere_hard", "cohere_det", "lgbm", "lgbm_det", "lgbm_soft"]:
                J.append((f"s0_{c}", m, 0, None))
        for tag in ["s0_c20", "s1_c20", "s2_c20"]:
            for m in ["lgbm_det", "lr_det", "lgbm_soft"]:
                J.append((tag, m, 0, None))
        for sd in [0, 1]:
            for m in ["cohere", "cohere_nopress"]:
                J.append(("s0_c20_null", m, sd, None))
        for ho in ["0,1", "2,3", "4,5", "6,8,9", "7"]:
            for m in ["cohere", "cohere_nopress"]:
                J.append(("s0_c20", m, 0, ho))
    return J

def done(j):
    tag, m, sd, ho = j
    ot = f"{tag}__{m}__{sd}" + (f"__ho{ho.replace(',', '-')}" if ho else "")
    return os.path.exists(f"{WORK}/runs/{ot}.pkl")

def run(j):
    if done(j):
        return
    tag, m, sd, ho = j
    while not os.path.exists(f"{WORK}/prep_{tag}.pkl"):  # base file is written last
        time.sleep(20)
    cmd = ["python3", "run_patient.py", tag, m, str(sd)] + ([ho] if ho else [])
    with open(f"{WORK}/runs_log.txt", "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    phase = int(sys.argv[1]); nw = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    os.makedirs(f"{WORK}/runs", exist_ok=True)
    with ThreadPoolExecutor(nw) as ex:
        list(ex.map(run, job_list(phase)))
    print("phase", phase, "complete")
