"""
sweep_engine.py

Parameter sweep engine for MAPbI3 perovskite solar cell project.

This file:
1. Runs one-parameter sweeps.
2. Saves sweep plots.
3. Generates a dataset for machine learning.

It uses the PerovskiteSolarCell class from solar_cell.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solar_cell import PerovskiteSolarCell


os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def run_sweep(param_name, values, base_params, x_label, filename, xscale="linear"):
    """
    Sweep one parameter while keeping all other parameters fixed.
    Saves a plot with Voc, Jsc, FF, and PCE.
    """

    records = []

    for val in values:
        params = base_params.copy()
        params[param_name] = val

        cell = PerovskiteSolarCell(**params)
        perf = cell.performance()

        records.append({
            param_name: val,
            "Voc": perf["Voc"],
            "Jsc": perf["Jsc"],
            "FF": perf["FF"],
            "PCE": perf["PCE"],
            "Vmp": perf["Vmp"],
            "Jmp": perf["Jmp"],
            "Pmax": perf["Pmax"],
        })

    df = pd.DataFrame(records)

    # Save sweep data
    df.to_csv(f"data/{filename}.csv", index=False)

    # Plot all four important metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    metrics = ["Voc", "Jsc", "FF", "PCE"]
    ylabels = ["Voc (V)", "Jsc (mA/cm²)", "FF (%)", "PCE (%)"]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        ax.plot(df[param_name], df[metric], marker="o", linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {x_label}")
        ax.grid(True, alpha=0.3)

        if xscale == "log":
            ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=300)
    plt.close()

    print(f"Saved: data/{filename}.csv and plots/{filename}.png")
    return df


def run_all_sweeps():
    """
    Run all important parameter sweeps.
    """

    base_params = {
        "Jsc_ref": 30.0,
        "J0_ref": 5e-10,
        "n": 1.5,
        "Rs": 2.0,
        "Rsh": 3000.0,
        "T": 300.0,
        "bandgap": 1.55,
        "thickness": 350.0,
        "bulk_Nt": 1e13,
        "iface_Nt": 1e11,
    }

    print("\nRunning thickness sweep...")
    run_sweep(
        param_name="thickness",
        values=np.linspace(100, 700, 25),
        base_params=base_params,
        x_label="Absorber thickness (nm)",
        filename="sweep_thickness",
    )

    print("\nRunning bulk defect density sweep...")
    run_sweep(
        param_name="bulk_Nt",
        values=np.logspace(11, 17, 25),
        base_params=base_params,
        x_label="Bulk defect density Nt (cm⁻³)",
        filename="sweep_bulk_Nt",
        xscale="log",
    )

    print("\nRunning interface defect density sweep...")
    run_sweep(
        param_name="iface_Nt",
        values=np.logspace(9, 16, 25),
        base_params=base_params,
        x_label="Interface defect density Nt (cm⁻³)",
        filename="sweep_iface_Nt",
        xscale="log",
    )

    print("\nRunning series resistance sweep...")
    run_sweep(
        param_name="Rs",
        values=np.linspace(0.2, 12, 25),
        base_params=base_params,
        x_label="Series resistance Rs (Ω cm²)",
        filename="sweep_Rs",
    )

    print("\nRunning shunt resistance sweep...")
    run_sweep(
        param_name="Rsh",
        values=np.logspace(2.5, 4.5, 25),
        base_params=base_params,
        x_label="Shunt resistance Rsh (Ω cm²)",
        filename="sweep_Rsh",
        xscale="log",
    )

    print("\nRunning temperature sweep...")
    run_sweep(
        param_name="T",
        values=np.linspace(280, 380, 21),
        base_params=base_params,
        x_label="Temperature (K)",
        filename="sweep_temperature",
    )

    print("\nRunning bandgap sweep...")
    run_sweep(
        param_name="bandgap",
        values=np.linspace(1.40, 1.90, 25),
        base_params=base_params,
        x_label="Bandgap Eg (eV)",
        filename="sweep_bandgap",
    )

    print("\nRunning ideality factor sweep...")
    run_sweep(
        param_name="n",
        values=np.linspace(1.0, 2.0, 21),
        base_params=base_params,
        x_label="Ideality factor n",
        filename="sweep_ideality_factor",
    )


def build_ml_dataset(n_random=800):
    """
    Generate dataset using random sampling.
    This dataset is used for ML surrogate modelling.
    """

    rng = np.random.default_rng(42)
    records = []

    for _ in range(n_random):
        params = {
            "Jsc_ref": rng.uniform(26, 32),
            "J0_ref": 10 ** rng.uniform(-11, -8),
            "n": rng.uniform(1.0, 2.0),
            "Rs": rng.uniform(0.2, 10),
            "Rsh": 10 ** rng.uniform(2.5, 4.5),
            "T": rng.uniform(280, 380),
            "bandgap": rng.uniform(1.40, 1.90),
            "thickness": rng.uniform(100, 700),
            "bulk_Nt": 10 ** rng.uniform(11, 17),
            "iface_Nt": 10 ** rng.uniform(9, 16),
        }

        cell = PerovskiteSolarCell(**params)
        perf = cell.performance()

        row = {
            **params,
            "Voc": perf["Voc"],
            "Jsc": perf["Jsc"],
            "FF": perf["FF"],
            "PCE": perf["PCE"],
            "Vmp": perf["Vmp"],
            "Jmp": perf["Jmp"],
            "Pmax": perf["Pmax"],
        }

        records.append(row)

    df = pd.DataFrame(records)

    # Remove clearly non-useful or failed points
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["PCE"] > 0]
    df = df[df["PCE"] < 35]

    df.to_csv("data/perovskite_dataset.csv", index=False)

    print("\nML dataset generated.")
    print(f"Samples: {len(df)}")
    print("Saved: data/perovskite_dataset.csv")

    return df


if __name__ == "__main__":
    run_all_sweeps()
    build_ml_dataset(n_random=800)