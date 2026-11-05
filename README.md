# COHERE-Resp

Linkage-uncertainty-aware federated learning that couples fragmented patient records with wastewater
epidemic pressure for coherent deterioration and oxygen-demand forecasting in respiratory infections.

This folder contains the complete, flat implementation used for the manuscript: cohort generator driven
by real wastewater, Bloom-filter privacy-preserving linkage with null-normalised membership probabilities,
the membership-attention patient module with gated pressure cross-attention, discrete-time hazards,
patient-to-facility MinT reconciliation, federated training with pairwise-mask secure aggregation,
weighted conformal calibration, and the analysis and figure scripts.

## Files

| File | Purpose |
|---|---|
| `config.py` | Sites, study period, splits, variant transitions |
| `pressure.py` | Real wastewater features (PHAC sewersheds, Ottawa influenza/RSV), non-detect handling |
| `simulate.py` | Ten-site cohort generator: admissions, physiology, fragmentation, identifier corruption |
| `linkage.py` | Bloom-filter encoding, Dice features, calibrated odds, null-normalised membership, baselines |
| `dataset.py` | Landmark construction and token sets under each linkage policy |
| `models.py` | COHERE-Resp patient module, GRU-D, summary features, survival likelihood |
| `train.py` | Pooled, federated (FedAvg/FedProx) training, secure aggregation, coherence loss |
| `facility.py` | Retention curves, bottom-up/direct forecasts, MinT reconciliation |
| `conformal.py` | Split, covariate-shift weighted and adaptive conformal procedures |
| `prep.py`, `run_patient.py`, `queue_runs.py` | Cohort preparation and experiment runner |
| `tier_b_provinces.py` | Real provincial admissions vs real wastewater |
| `tier_c_physionet.py` | Real ICU time series (PhysioNet/CinC 2019), fragmented and linked |
| `analysis.py`, `figs_results.py`, `fig_architecture.py` | Results, statistics, figures |
| `download_data.py` | Fetches the public input data |
| `run_all.sh` | Reproduces everything |

## Quick start

```bash
pip install -r requirements.txt
python3 download_data.py
bash run_all.sh
```

Outputs are written to `work/` (cached cohorts, runs, results) and `figs/`.

## Data

All inputs are public: CovidTimelineCanada (CC BY 4.0), PHESD (CC BY 4.0), PhysioNet/CinC 2019
challenge data. No identifiable information is used; identifiers in Tier A and Tier C are synthetic.

## License

MIT (code). Data remain under their original licences.
