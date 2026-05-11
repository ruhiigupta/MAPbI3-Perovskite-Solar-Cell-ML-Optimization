"""
inspect_data.py

Quick inspection of the generated perovskite solar-cell dataset.
This checks ranges, missing values, and basic correlations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/perovskite_dataset.csv")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics for outputs:")
print(df[["Voc", "Jsc", "FF", "PCE"]].describe())

print("\nPCE range:")
print("Minimum PCE:", df["PCE"].min())
print("Maximum PCE:", df["PCE"].max())
print("Mean PCE:", df["PCE"].mean())

# Histogram of PCE
plt.figure(figsize=(7, 5))
plt.hist(df["PCE"], bins=30, edgecolor="black", alpha=0.8)
plt.xlabel("PCE (%)")
plt.ylabel("Count")
plt.title("Distribution of PCE in Generated Dataset")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/pce_distribution.png", dpi=300)
plt.show()

# Correlation heatmap
selected_cols = [
    "thickness",
    "bulk_Nt",
    "iface_Nt",
    "Rs",
    "Rsh",
    "T",
    "bandgap",
    "n",
    "Voc",
    "Jsc",
    "FF",
    "PCE",
]

corr = df[selected_cols].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Input Parameters and Outputs")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.show()