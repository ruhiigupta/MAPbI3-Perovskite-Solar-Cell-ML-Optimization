"""
sq_limit.py

Approximate Shockley-Queisser limit calculator.

This module estimates the theoretical maximum PCE as a function of bandgap.
It is used to compare:
- silicon
- MAPbI3
- wide-bandgap perovskite for tandem cells

This is a project-level approximation, not a full detailed-balance solver.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def approximate_sq_limit(Eg):
    """
    Approximate Shockley-Queisser PCE curve as a function of bandgap.

    Peak is around Eg ~ 1.34 eV with max PCE ~33.7%.
    MAPbI3 at Eg ~1.55 eV gives around 31-33%.
    """

    Eg_opt = 1.34
    max_pce = 33.7

    # Main peak around optimum bandgap
    pce = max_pce * np.exp(-0.55 * (Eg - Eg_opt) ** 2)

    # Efficiency falls at high bandgap because Jsc decreases
    if Eg > 1.6:
        pce *= np.exp(-0.35 * (Eg - 1.6))

    # Efficiency falls at low bandgap because voltage decreases
    if Eg < 1.1:
        pce *= np.exp(-0.8 * (1.1 - Eg))

    return pce


def plot_sq_limit():
    os.makedirs("plots", exist_ok=True)

    Eg_values = np.linspace(0.7, 2.4, 300)
    pce_values = np.array([approximate_sq_limit(Eg) for Eg in Eg_values])

    materials = {
        "Si": 1.12,
        "MAPbI3": 1.55,
        "Wide-Eg PSC": 1.80,
    }

    plt.figure(figsize=(8, 5))
    plt.plot(Eg_values, pce_values, linewidth=2.5, label="Approx. SQ limit")

    for name, Eg in materials.items():
        pce = approximate_sq_limit(Eg)
        plt.scatter(Eg, pce, s=70)
        plt.axvline(Eg, linestyle="--", alpha=0.5)
        plt.text(
            Eg + 0.02,
            pce - 2.0,
            f"{name}\nEg={Eg:.2f} eV\n{pce:.1f}%",
            fontsize=9,
        )

    plt.xlabel("Bandgap $E_g$ (eV)")
    plt.ylabel("Theoretical maximum PCE (%)")
    plt.title("Shockley–Queisser Limit vs Bandgap")
    plt.ylim(0, 38)
    plt.xlim(0.7, 2.4)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sq_limit.png", dpi=300)
    plt.show()

    mapbi3_pce = approximate_sq_limit(1.55)

    print("\nApproximate Shockley-Queisser Result")
    print("------------------------------------")
    print("MAPbI3 bandgap: 1.55 eV")
    print(f"Approx. SQ limit PCE: {mapbi3_pce:.2f} %")


if __name__ == "__main__":
    plot_sq_limit()