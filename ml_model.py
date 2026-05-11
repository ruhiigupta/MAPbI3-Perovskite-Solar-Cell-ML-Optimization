"""
ml_model.py

Machine learning surrogate model for MAPbI3 perovskite solar-cell performance.

This file:
1. Loads the generated dataset.
2. Trains several ML models to predict PCE.
3. Compares R2, MAE, and RMSE.
4. Saves model comparison and feature importance plots.
5. Saves the best model.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv("data/perovskite_dataset.csv")

print("\nLoaded dataset:")
print(df.shape)

# Log-transform features that vary over many orders of magnitude
df["log_J0_ref"] = np.log10(df["J0_ref"])
df["log_bulk_Nt"] = np.log10(df["bulk_Nt"])
df["log_iface_Nt"] = np.log10(df["iface_Nt"])
df["log_Rsh"] = np.log10(df["Rsh"])

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

target_col = "PCE"

X = df[feature_cols]
y = df[target_col]

print("\nFeatures used:")
print(feature_cols)

print("\nTarget:")
print(target_col)


# ------------------------------------------------------------
# 2. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# ------------------------------------------------------------
# 3. Define models
# ------------------------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        objective="reg:squarederror"
    )


# ------------------------------------------------------------
# 4. Train and evaluate models
# ------------------------------------------------------------

results = []
best_model = None
best_name = None
best_r2 = -999

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
    })

    print(f"R2   = {r2:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name


results_df = pd.DataFrame(results)
results_df.to_csv("data/model_results.csv", index=False)

print("\nModel comparison:")
print(results_df)

print(f"\nBest model: {best_name}")
print(f"Best R2: {best_r2:.4f}")


# ------------------------------------------------------------
# 5. Plot model comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("R² score")
plt.title("Model Comparison for PCE Prediction")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/model_comparison_r2.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.ylabel("MAE in PCE (%)")
plt.title("Model Comparison: Mean Absolute Error")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/model_comparison_mae.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 6. Actual vs predicted plot for best model
# ------------------------------------------------------------

best_pred = best_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.7)
plt.xlabel("Actual PCE (%)")
plt.ylabel("Predicted PCE (%)")
plt.title(f"Actual vs Predicted PCE ({best_name})")

min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted_pce.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 7. Feature importance
# ------------------------------------------------------------

# Use Random Forest or XGBoost feature importance if available
if best_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    importances = best_model.feature_importances_
else:
    # For linear models, use absolute coefficients
    model_inside = best_model.named_steps["model"]
    importances = np.abs(model_inside.coef_)

feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

feature_importance.to_csv("data/feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Feature importance")
plt.title(f"Feature Importance for PCE Prediction ({best_name})")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 8. Save best model
# ------------------------------------------------------------

joblib.dump(best_model, "data/best_pce_model.pkl")

print("\nSaved:")
print("data/model_results.csv")
print("data/feature_importance.csv")
print("data/best_pce_model.pkl")
print("plots/model_comparison_r2.png")
print("plots/model_comparison_mae.png")
print("plots/actual_vs_predicted_pce.png")
print("plots/feature_importance.png")