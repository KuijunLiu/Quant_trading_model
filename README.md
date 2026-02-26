# Quant-Trading-Model
A monthly stock-selection pipeline using CRSP data for feature engineering, model training, and backtesting.

## Installation
- Python = 3.12
bash
conda env create -f environment.yml
conda activate quant_env


----------------------------------
## Project Structure
.
├── data/                  # (optional) local raw/processed data (ignored by git)
├── notebooks/             # exploration / experiments
├── src/                   # core code
│   ├── download_data/     # data loading & cleaning
│   ├── features/          # feature engineering
│   ├── models/            # training / inference
│   ├── backtest/          # portfolio construction & backtest
├── reports/               # figures/tables
├── tests/                 # unit tests
├── environment.yml        # create environment
└── README.md

-----------------------------------
## Data
This project expects CRSP monthly stock file (MSF) exported from WRDS.

Required fields:
- PERMNO, DATE, RET, PRC, SHROUT, EXCHCD, SHRCD

Place raw files under:
- data/raw/msf.parquet (or .csv)

⚠️ Do NOT commit CRSP data to this repository.
