#!/bin/bash
# Reproduce every experiment, table and figure (CPU only; roughly 5-6 hours on 2 cores).
set -e
cd "$(dirname "$0")"
python3 download_data.py
python3 tier_b_provinces.py                       # Tier B: real provincial admissions
python3 fig_architecture.py                       # Fig. 1
for s in 0 1 2; do python3 prep.py $s 0.20; done  # Tier A cohorts
python3 queue_runs.py 1 2                         # main comparison, ablations, federation, label delay
for c in 0.0 0.1 0.3 0.4; do python3 -c "import prep; prep.prepare(0,$c,modes=('soft','hard','det'))"; done
python3 -c "import prep; prep.prepare(0,0.2,null_pressure=True,modes=('soft',))"
python3 queue_runs.py 2 2                         # corruption sweep, null-pressure world, leave-city-out
python3 tier_c_physionet.py prep                  # Tier C: real ICU series
for s in 0 1 2; do python3 tier_c_physionet.py $s; done
python3 tier_c_physionet.py merge
python3 analysis.py                               # results object, statistics, Figs. 2-9
