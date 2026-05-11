"""
solar_cell.py

Physics-informed single-diode model for a MAPbI3 perovskite solar cell.

Device context:
FTO / TiO2 / MAPbI3 / Spiro-OMeTAD / Au

This is not a full SCAPS/drift-diffusion simulator.
It is a calibrated single-diode numerical model for:
- J-V curve
- Voc
- Jsc
- FF
- PCE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


Q = 1.602e-19
K_B = 1.381e-23
PIN = 100.0  # mW/cm^2


class PerovskiteSolarCell:
    def __init__(
        self,
        Jsc_ref=30.0,          # mA/cm^2
        J0_ref=5e-10,          # mA/cm^2
        n=1.5,
        Rs=2.0,                # ohm cm^2
        Rsh=3000.0,            # ohm cm^2
        T=300.0,
        bandgap=1.55,
        thickness=350.0,
        bulk_Nt=1e13,
        iface_Nt=1e11,
        alpha_nm_inv=1 / 300,
    ):
        self.Jsc_ref = Jsc_ref
        self.J0_ref = J0_ref
        self.n_input = n
        self.Rs = Rs
        self.Rsh = Rsh
        self.T = T
        self.bandgap = bandgap
        self.thickness = thickness
        self.bulk_Nt = bulk_Nt
        self.iface_Nt = iface_Nt
        self.alpha_nm_inv = alpha_nm_inv

        self.Vt = K_B * self.T / Q

        # Thickness-dependent photocurrent
        absorption_factor = 1 - np.exp(-self.alpha_nm_inv * self.thickness)
        self.Jsc = self.Jsc_ref * absorption_factor

        # Recombination dependence
        Nt_ref = 1e13
        iface_ref = 1e11

        bulk_factor = (self.bulk_Nt / Nt_ref) ** 0.5
        iface_factor = 1 + 0.35 * (self.iface_Nt / iface_ref) ** 0.25

        # Temperature dependence of dark current
        Eg_J = self.bandgap * Q
        temp_factor = (self.T / 300.0) ** 3 * np.exp(
            -(Eg_J / K_B) * ((1 / self.T) - (1 / 300.0))
        )

        self.J0 = self.J0_ref * bulk_factor * iface_factor * temp_factor

        # Ideality factor increases slightly with defect density
        defect_ratio = max(self.bulk_Nt / Nt_ref, 1.0)
        self.n = self.n_input + 0.08 * np.log10(defect_ratio)

    def J_at_V(self, V):
        """
        Solves the implicit single-diode equation.

        J is in mA/cm^2.
        Rs is in ohm cm^2.
        So the series voltage drop is:
        (J / 1000) * Rs
        """

        def residual(J):
            voltage_internal = V + (J / 1000.0) * self.Rs

            # avoid overflow during sweeps
            exponent = voltage_internal / (self.n * self.Vt)
            exponent = np.clip(exponent, -100, 100)

            diode_current = self.J0 * (np.exp(exponent) - 1)

            # shunt current: A/cm^2 converted to mA/cm^2
            shunt_current = (voltage_internal / self.Rsh) * 1000.0

            return J - self.Jsc + diode_current + shunt_current

        try:
            return brentq(residual, -100.0, 100.0, xtol=1e-10)
        except ValueError:
            return np.nan

    def JV_curve(self, V_min=0.0, V_max=1.3, n_points=500):
        V = np.linspace(V_min, V_max, n_points)
        J = np.array([self.J_at_V(v) for v in V])

        # replace failed points with 0 only after useful region
        J = np.nan_to_num(J, nan=0.0)
        return V, J

    def performance(self):
        V, J = self.JV_curve()

        Jsc_out = self.J_at_V(0.0)

        Voc = 0.0
        for i in range(len(V) - 1):
            if J[i] >= 0 and J[i + 1] <= 0:
                try:
                    Voc = brentq(self.J_at_V, V[i], V[i + 1])
                except ValueError:
                    Voc = V[i]
                break

        power = V * J  # mW/cm^2
        valid = (V > 0) & (J > 0)

        if np.any(valid):
            power_valid = np.where(valid, power, -np.inf)
            idx = np.argmax(power_valid)
            Pmax = power[idx]
            Vmp = V[idx]
            Jmp = J[idx]
        else:
            Pmax = 0.0
            Vmp = 0.0
            Jmp = 0.0

        FF = (Pmax / (Voc * Jsc_out)) * 100 if Voc > 0 and Jsc_out > 0 else 0.0
        PCE = (Pmax / PIN) * 100

        return {
            "Voc": Voc,
            "Jsc": Jsc_out,
            "FF": FF,
            "PCE": PCE,
            "Vmp": Vmp,
            "Jmp": Jmp,
            "Pmax": Pmax,
        }

    def plot_JV(self, label="Baseline MAPbI3", filename="baseline_JV.png"):
        os.makedirs("plots", exist_ok=True)

        V, J = self.JV_curve()
        perf = self.performance()

        plt.figure(figsize=(7, 5))
        plt.plot(
            V,
            J,
            linewidth=2,
            label=f"{label}: PCE={perf['PCE']:.2f}%, Voc={perf['Voc']:.3f} V",
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylim(-5, self.Jsc + 5)
        plt.xlim(0, 1.15)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current density J (mA/cm²)")
        plt.title("Baseline Perovskite Solar Cell J–V Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("plots", filename), dpi=300)
        plt.show()


if __name__ == "__main__":
    cell = PerovskiteSolarCell()
    perf = cell.performance()

    print("\nBaseline MAPbI3 Perovskite Solar Cell")
    print("-------------------------------------")
    for key, value in perf.items():
        print(f"{key:5s}: {value:.4f}")

    cell.plot_JV()