"""Download the public input data into this folder.

Sources (all open):
  phac_ww.csv, hospitalizations_pt.csv, hosp_admissions_pt.csv - COVID-19 Canada Open Data Working Group,
      CovidTimelineCanada (CC BY 4.0), https://github.com/ccodwg/CovidTimelineCanada
  ottawa_ww.csv - PHESD, Delatolla Lab, University of Ottawa (CC BY 4.0), https://github.com/Big-Life-Lab/PHESD
  sepsis.parquet - PhysioNet/CinC Challenge 2019 training data (open access), Hugging Face mirror
"""
import os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CTC = "https://raw.githubusercontent.com/ccodwg/CovidTimelineCanada/main"
FILES = {
    "phac_ww.csv": f"{CTC}/extra_data/phac_wastewater/phac_wastewater.csv",
    "hospitalizations_pt.csv": f"{CTC}/data/pt/hospitalizations_pt.csv",
    "hosp_admissions_pt.csv": f"{CTC}/data/pt/hosp_admissions_pt.csv",
    "ottawa_ww.csv": "https://raw.githubusercontent.com/Big-Life-Lab/PHESD/main/Wastewater/Ottawa/Data/wastewater_virus.csv",
    "sepsis.parquet": "https://huggingface.co/datasets/legomaheggo/physionet-sepsis-2019/resolve/main/data/train-00000-of-00001.parquet",
}

if __name__ == "__main__":
    for name, url in FILES.items():
        dst = os.path.join(HERE, name)
        if os.path.exists(dst) and not os.path.islink(dst):
            print("exists", name); continue
        print("downloading", name)
        urllib.request.urlretrieve(url, dst)
    print("done")
