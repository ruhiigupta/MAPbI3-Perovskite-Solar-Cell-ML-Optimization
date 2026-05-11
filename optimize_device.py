"""
optimize_device.py

ML-guided optimization of MAPbI3 perovskite solar-cell parameters.

This file:
1. Loads the trained best ML model.
2. Searches possible device parameters.
3. Predicts the best PCE using ML.
4. Verifies the optimized design using the physics simulator.
5. Plots baseline vs optimized J-V curve.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from solar_cell import PerovskiteSolarCell


os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------
# 1. Load trained model
# ------------------------------------------------------------

model = joblib.load("data/best_pce_model.pkl")


# Feature order must match ml_model.py
feature_cols = [
    "Jsc_ref",
    "log_J0_ref",
    "n",
    "Rs",
    "log_Rsh",
    "T",
    "bandgap",
    "thickness",
    "log_bulk_Nt",
    "log_iface_Nt",
]


# ------------------------------------------------------------
# 2. Define objective function
# ------------------------------------------------------------

def negative_pce(x):
    """
    x contains:
    Jsc_ref, log_J0_ref, n, Rs, log_Rsh, T, bandgap,
    thickness, log_bulk_Nt, log_iface_Nt

    We minimize negative PCE, which means maximizing PCE.
    """

    x_df = pd.DataFrame([x], columns=feature_cols)
    predicted_pce = model.predict(x_df)[0]
    return -predicted_pce


# ------------------------------------------------------------
# 3. Parameter bounds
# ------------------------------------------------------------

bounds = [
    (28.0, 32.0),    # Jsc_ref
    (-11.0, -9.0),   # log_J0_ref
    (1.0, 1.5),      # ideality factor n
    (0.2, 3.0),      # Rs
    (3.3, 4.5),      # log_Rsh
    (295.0, 305.0),  # T near room temperature
    (1.50, 1.65),    # bandgap
    (250.0, 700.0),  # absorber thickness
    (11.0, 13.0),    # log_bulk_Nt
    (9.0, 11.0),     # log_iface_Nt
]


# ------------------------------------------------------------
# 4. Run differential evolution optimization
# ------------------------------------------------------------

result = differential_evolution(
    negative_pce,
    bounds,
    seed=42,
    maxiter=150,
    polish=True,
)

opt_x = result.x
predicted_opt_pce = -result.fun

opt_params_ml = dict(zip(feature_cols, opt_x))

print("\nML-optimized parameters")
print("-----------------------")
for key, value in opt_params_ml.items():
    print(f"{key:15s}: {value:.4f}")

print(f"\nML-predicted optimized PCE: {predicted_opt_pce:.4f} %")


# ------------------------------------------------------------
# 5. Convert ML parameters back to physics simulator parameters
# ------------------------------------------------------------

physics_params = {
    "Jsc_ref": opt_params_ml["Jsc_ref"],
    "J0_ref": 10 ** opt_params_ml["log_J0_ref"],
    "n": opt_params_ml["n"],
    "Rs": opt_params_ml["Rs"],
    "Rsh": 10 ** opt_params_ml["log_Rsh"],
    "T": opt_params_ml["T"],
    "bandgap": opt_params_ml["bandgap"],
    "thickness": opt_params_ml["thickness"],
    "bulk_Nt": 10 ** opt_params_ml["log_bulk_Nt"],
    "iface_Nt": 10 ** opt_params_ml["log_iface_Nt"],
}


# ------------------------------------------------------------
# 6. Verify with physics simulator
# ------------------------------------------------------------

baseline_cell = PerovskiteSolarCell()
optimized_cell = PerovskiteSolarCell(**physics_params)

baseline_perf = baseline_cell.performance()
optimized_perf = optimized_cell.performance()

print("\nBaseline performance")
print("--------------------")
for key, value in baseline_perf.items():
    print(f"{key:5s}: {value:.4f}")

print("\nOptimized performance from physics model")
print("----------------------------------------")
for key, value in optimized_perf.items():
    print(f"{key:5s}: {value:.4f}")

delta_pce = optimized_perf["PCE"] - baseline_perf["PCE"]

print(f"\nDelta PCE = {delta_pce:.4f} %")


# ------------------------------------------------------------
# 7. Save optimized parameter table
# ------------------------------------------------------------

opt_table = pd.DataFrame([
    {"Parameter": "Jsc_ref", "Baseline": 30.0, "Optimized": physics_params["Jsc_ref"]},
    {"Parameter": "J0_ref", "Baseline": 5e-10, "Optimized": physics_params["J0_ref"]},
    {"Parameter": "n", "Baseline": 1.5, "Optimized": physics_params["n"]},
    {"Parameter": "Rs", "Baseline": 2.0, "Optimized": physics_params["Rs"]},
    {"Parameter": "Rsh", "Baseline": 3000.0, "Optimized": physics_params["Rsh"]},
    {"Parameter": "T", "Baseline": 300.0, "Optimized": physics_params["T"]},
    {"Parameter": "bandgap", "Baseline": 1.55, "Optimized": physics_params["bandgap"]},
    {"Parameter": "thickness", "Baseline": 350.0, "Optimized": physics_params["thickness"]},
    {"Parameter": "bulk_Nt", "Baseline": 1e13, "Optimized": physics_params["bulk_Nt"]},
    {"Parameter": "iface_Nt", "Baseline": 1e11, "Optimized": physics_params["iface_Nt"]},
])

opt_table.to_csv("data/optimized_parameters.csv", index=False)


performance_table = pd.DataFrame([
    {"Metric": "Voc", "Baseline": baseline_perf["Voc"], "Optimized": optimized_perf["Voc"]},
    {"Metric": "Jsc", "Baseline": baseline_perf["Jsc"], "Optimized": optimized_perf["Jsc"]},
    {"Metric": "FF", "Baseline": baseline_perf["FF"], "Optimized": optimized_perf["FF"]},
    {"Metric": "PCE", "Baseline": baseline_perf["PCE"], "Optimized": optimized_perf["PCE"]},
])

performance_table.to_csv("data/baseline_vs_optimized_performance.csv", index=False)


# ------------------------------------------------------------
# 8. Plot baseline vs optimized J-V curve
# ------------------------------------------------------------

V_base, J_base = baseline_cell.JV_curve()
V_opt, J_opt = optimized_cell.JV_curve()

plt.figure(figsize=(7, 5))
plt.plot(
    V_base,
    J_base,
    linewidth=2,
    label=f"Baseline: PCE={baseline_perf['PCE']:.2f}%",
)
plt.plot(
    V_opt,
    J_opt,
    linewidth=2,
    linestyle="--",
    label=f"Optimized: PCE={optimized_perf['PCE']:.2f}%",
)

plt.axhline(0, color="black", linewidth=0.8)
plt.ylim(-5, max(baseline_perf["Jsc"], optimized_perf["Jsc"]) + 5)
plt.xlim(0, 1.2)
plt.xlabel("Voltage (V)")
plt.ylabel("Current density J (mA/cm²)")
plt.title("Baseline vs ML-Optimized J–V Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/optimized_JV.png", dpi=300)
plt.show()


print("\nSaved:")
print("data/optimized_parameters.csv")
print("data/baseline_vs_optimized_performance.csv")
print("plots/optimized_JV.png")