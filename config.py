"""Shared configuration for the COHERE-Resp experiments."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("COHERE_RAW", ROOT)
WORK = os.environ.get("COHERE_WORK", os.path.join(ROOT, "work"))
os.makedirs(WORK, exist_ok=True)

# Ten hospital sites, each served by one real PHAC sewershed
SITES = [
    "Toronto Ashbridges Bay", "Toronto Humber", "Montreal North", "Montreal South",
    "Vancouver Annacis Island", "Vancouver Iona Island", "Winnipeg North End",
    "Halifax Halifax", "Edmonton Goldbar", "Saskatoon",
]
START, END = "2021-05-01", "2023-12-31"
TRAIN_END, CAL_END = "2023-01-31", "2023-05-31"   # test = Jun-Dec 2023
REPORT_DELAY = 3          # days from wastewater sampling to availability
PRESSURE_WINDOW = 21      # days of pressure history seen by the model
LANDMARKS = (12, 36)      # hours after admission at which predictions are issued
HORIZON_DAYS = 7          # discrete-time hazard bins (days)
LOOKBACK_H = 72           # hours of evidence before admission eligible for linkage
MAX_TOKENS = 48

# Variant era transitions (logistic midpoints) and relative progression odds
VARIANTS = ["Alpha", "Delta", "BA.1", "BA.2", "BA.5", "XBB"]
VARIANT_MID = ["2021-07-01", "2021-12-20", "2022-03-25", "2022-07-01", "2023-01-25"]
VARIANT_SEV = [0.85, 1.00, 0.55, 0.55, 0.60, 0.55]
