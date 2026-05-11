# Python-Based Physics-Informed Simulation and ML-Assisted Optimization of MAPbI3 Perovskite Solar Cells

This project presents a Python-based physics-informed simulation and machine learning-assisted optimization framework for a planar MAPbI3 perovskite solar cell with the device stack:

FTO / TiO2 / MAPbI3 / Spiro-OMeTAD / Au

The project combines a calibrated single-diode solar-cell model, Shockley–Queisser limit analysis, parameter sweeps, machine learning surrogate modelling, SHAP explainability, and ML-guided optimization.

## Project Highlights

- Baseline simulated PCE: 15.13%
- Approximate Shockley–Queisser limit for MAPbI3: 32.89%
- Best ML model: Gradient Boosting
- Best model R2 score: 0.9589
- Best model MAE: 0.8554% PCE
- Optimized simulated PCE: 27.17%
- PCE improvement: +12.05 percentage points

## Device Structure

FTO / TiO2 / MAPbI3 / Spiro-OMeTAD / Au

## Python Modules

| File | Purpose |
|---|---|
| solar_cell.py | Physics-informed single-diode simulator |
| sq_limit.py | Shockley–Queisser limit analysis |
| sweep_engine.py | Parameter sweeps and dataset generation |
| inspect_data.py | Dataset inspection and correlation analysis |
| ml_model.py | ML model training and comparison |
| shap_analysis.py | SHAP explainability analysis |
| optimize_device.py | ML-guided device optimization |
| summary_tables.py | Final report summary tables |

## Results

### Baseline Device

| Metric | Value |
|---|---:|
| Voc | 0.9360 V |
| Jsc | 20.6441 mA/cm2 |
| FF | 78.28% |
| PCE | 15.13% |

### Optimized Device

| Metric | Value |
|---|---:|
| Voc | 1.1677 V |
| Jsc | 27.3270 mA/cm2 |
| FF | 85.16% |
| PCE | 27.17% |

## ML Model Performance

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.9118 | 1.2017 | 1.5812 |
| Ridge Regression | 0.9118 | 1.2013 | 1.5807 |
| Random Forest | 0.9063 | 1.2931 | 1.6293 |
| Gradient Boosting | 0.9589 | 0.8554 | 1.0795 |
| XGBoost | 0.9566 | 0.8755 | 1.1094 |

## How to Run

Install dependencies:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn xgboost shap joblib