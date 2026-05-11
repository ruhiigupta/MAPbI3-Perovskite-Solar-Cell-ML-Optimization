"""
summary_tables.py

Creates clean summary tables for the final report:
1. Baseline performance
2. Optimized performance
3. Baseline vs optimized comparison
4. Key project results
"""

import os
import pandas as pd

os.makedirs("data", exist_ok=True)

baseline = {
    "Voc (V)": 0.9360,
    "Jsc (mA/cm²)": 20.6441,
    "FF (%)": 78.2842,
    "PCE (%)": 15.1272,
}

optimized = {
    "Voc (V)": 1.1677,
    "Jsc (mA/cm²)": 27.3270,
    "FF (%)": 85.1598,
    "PCE (%)": 27.1743,
}

comparison_rows = []

for metric in baseline:
    base_val = baseline[metric]
    opt_val = optimized[metric]
    improvement = opt_val - base_val
    percent_improvement = (improvement / base_val) * 100

    comparison_rows.append({
        "Metric": metric,
        "Baseline": base_val,
        "Optimized": opt_val,
        "Absolute improvement": improvement,
        "Percentage improvement (%)": percent_improvement,
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("data/final_baseline_vs_optimized.csv", index=False)

key_results = pd.DataFrame([
    {
        "Result": "Baseline simulated PCE",
        "Value": "15.13 %",
    },
    {
        "Result": "Approximate SQ limit for MAPbI3",
        "Value": "32.89 %",
    },
    {
        "Result": "Best ML model",
        "Value": "Gradient Boosting",
    },
    {
        "Result": "Best model R²",
        "Value": "0.9589",
    },
    {
        "Result": "Best model MAE",
        "Value": "0.8554 % PCE",
    },
    {
        "Result": "Optimized simulated PCE",
        "Value": "27.17 %",
    },
    {
        "Result": "PCE improvement",
        "Value": "12.05 percentage points",
    },
    {
        "Result": "Dominant SHAP features",
        "Value": "Temperature, absorber thickness, ideality factor, J0, series resistance",
    },
])

key_results.to_csv("data/key_project_results.csv", index=False)

print("\nFinal baseline vs optimized table:")
print(comparison_df)

print("\nKey project results:")
print(key_results)

print("\nSaved:")
print("data/final_baseline_vs_optimized.csv")
print("data/key_project_results.csv")