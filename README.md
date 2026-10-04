# PV Power Forecasting under Fog Weather Conditions

Reproducible experiment code and data for photovoltaic (PV) power forecasting under fog-weather conditions.

The main method combines **feature enhancement**, **adaptive asymmetric sample weighting**, and **ExtraTrees regression**, and compares the proposed approach with operational business forecasts and several machine-learning baselines.

## Repository structure

```text
fog-pv-forecasting/
├── data/
│   └── 78101000_next_day_data.csv
├── figure_table_v2.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Main experiments

The script produces:

- Overall model comparison
- Adaptive asymmetric weighting parameter search
- Ablation study
- Performance under different output-level fog conditions
- Feature importance before and after feature engineering
- Best-day prediction comparison
- Absolute-error and error-improvement distributions
- Multi-metric ablation improvement figure
- Overall ACC comparison

## Method overview

The workflow includes:

1. Time-feature construction
2. Solar-position feature construction
3. Business-forecast feature fusion
4. Fog-segment-aware difference and rolling features
5. Physical constraints for PV output
6. Adaptive asymmetric sample weighting
7. ExtraTrees / CatBoost / LightGBM / XGBoost / RandomForest comparison
8. Chronological 80/20 train-test split

The asymmetric sample weight is implemented as:

```text
w = 1 + alpha * (1 - y/cap)^power
```

where `alpha` and `power` are selected according to validation ACC.

## Dataset

The included CSV contains timestamped PV power, installed capacity, operational forecast power, and meteorological observation/forecast variables.

Important columns include:

- `time`: timestamp
- `cap`: installed PV capacity
- `obs_power`: observed PV power
- `pred_power`: operational/business forecast power
- meteorological variables such as temperature, humidity, pressure, wind and irradiance

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python figure_table_v2.py
```

All generated results are written to:

```text
outputs/
├── figures/
├── tables/
├── predictions/
└── experiment_explanation.txt
```

## Notes

- The code uses relative paths, so it can run after cloning the repository without editing local Windows paths.
- `pvlib` is used for solar-position calculation when available; the script contains a fallback approximate calculation.
- Nighttime PV predictions are set to zero and all predictions are constrained to `[0, installed capacity]`.

## Citation

If this repository is used to support an academic publication, please cite the corresponding paper once its bibliographic information is available.
