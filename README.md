# AssayingAnomalies-Python (work-in-progress)

Goal: a faithful Python port of the **Assaying Anomalies** MATLAB toolkit (Novy-Marx & Velikov), starting with CRSP monthly 1962–present, then expanding.  
Official site / repo describe the protocol and MATLAB implementation.  
- Overview & protocol: https://sites.psu.edu/assayinganomalies/  
- MATLAB repo: https://github.com/velikov-mihail/AssayingAnomalies

## Quickstart
```bash
py -3.11 -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -U pip -r requirements.txt
pre-commit install
python scripts/setup_library.py           # downloads & caches CRSP monthly via WRDS
python scripts/use_library.py             # runs toy signal (log size) through protocol
