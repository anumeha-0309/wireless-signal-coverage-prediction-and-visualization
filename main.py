# Major Project: ML-Based Cellular Signal Strength Prediction
# Focus: Regression to predict RSRP from geospatial features (Latitude, Longitude, Elevation)
# Tools: Pandas, Scikit-learn, XGBoost, Matplotlib, SciPy
# Steps: Load data → Prep features → Train model → Evaluate → Visualize

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("preprocessed_main_dataset_3.csv")

df["lat_lon_interact"] = df["Latitude"] * df["Longitude"]

features = ["Latitude", "Longitude", "Elevation", "lat_lon_interact"]
X = df[features]
y = df["RSRP"]

print(f"Dataset shape: {df.shape}")
print(f"Features: {features}")
print(f"RSRP Range: {y.min():.0f} to {y.max():.0f} dBm")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f} | R²: {r2_rf:.3f}")

importances_rf = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

# XGBoost
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Results:")
print(f"MAE: {mae_xgb:.2f} | RMSE: {rmse_xgb:.2f} | R²: {r2_xgb:.3f}")

importances_xgb = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

# SVR
svr = SVR(kernel="rbf", C=100, gamma=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

print("\nSVR Results:")
print(f"MAE: {mae_svr:.2f} | RMSE: {rmse_svr:.2f} | R²: {r2_svr:.3f}")

# KNN
knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
r2_knn = r2_score(y_test, y_pred_knn)

print("\nKNN Results:")
print(f"MAE: {mae_knn:.2f} | RMSE: {rmse_knn:.2f} | R²: {r2_knn:.3f}")

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gbr.fit(X_train_scaled, y_train)
y_pred_gbr = gbr.predict(X_test_scaled)

mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)

print("\nGradient Boosting Results:")
print(f"MAE: {mae_gbr:.2f} | RMSE: {rmse_gbr:.2f} | R²: {r2_gbr:.3f}")

models = {
    "Random Forest": rf,
    "XGBoost": xgb_model,
    "SVR": svr,
    "KNN": knn,
    "Gradient Boosting": gbr
}

cv_scores = {}
for name, model in models.items():
    cv_mae = -cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_scores[name] = cv_mae.mean()
    print(f"\n{name} CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f} dBm")

model_names = list(models.keys())

pred_dict = {
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn,
    "Gradient Boosting": y_pred_gbr
}

comparison = pd.DataFrame({
    "Method": model_names,
    "MAE (dBm)": [mae_rf, mae_xgb, mae_svr, mae_knn, mae_gbr],
    "RMSE (dBm)": [rmse_rf, rmse_xgb, rmse_svr, rmse_knn, rmse_gbr],
    "R²": [r2_rf, r2_xgb, r2_svr, r2_knn, r2_gbr]
})

print("\n--- Comparison ---")
print(comparison)

fig1, axes1 = plt.subplots(1, 5, figsize=(25, 5))
r2_dict = {"Random Forest": r2_rf, "XGBoost": r2_xgb, "SVR": r2_svr, "KNN": r2_knn, "Gradient Boosting": r2_gbr}

for i, name in enumerate(model_names):
    y_pred = pred_dict[name]
    axes1[i].scatter(y_test, y_pred, alpha=0.6)
    axes1[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes1[i].set_title(f"{name}\nR² = {r2_dict[name]:.3f}")

plt.tight_layout()
plt.savefig("actual_vs_pred_all_models.png", dpi=300)
plt.show()

fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5))
mae_dict = {'Random Forest': mae_rf, 'XGBoost': mae_xgb, 'SVR': mae_svr, 'KNN': mae_knn, 'Gradient Boosting': mae_gbr}

for i, name in enumerate(model_names):
    y_pred = pred_dict[name]
    residuals = y_test - y_pred
    axes2[i].scatter(y_pred, residuals, alpha=0.6)
    axes2[i].axhline(y=0, color='r', linestyle='--')
    axes2[i].set_xlabel('Predicted RSRP (dBm)')
    axes2[i].set_ylabel('Residuals (dBm)')
    axes2[i].set_title(f'{name}\n(MAE = {mae_dict[name]:.2f})')

plt.tight_layout()
plt.savefig('residuals_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

fig3, ax3 = plt.subplots(1, 3, figsize=(15, 6))
metrics = ['MAE (dBm)', 'RMSE (dBm)', 'R²']

for j, metric in enumerate(metrics):
    values = [comparison[metric].iloc[k] for k in range(len(model_names))]
    ax3[j].bar(model_names, values, alpha=0.7)
    ax3[j].set_title(metric)
    ax3[j].set_ylabel(metric)
    ax3[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('metrics_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()

tree_models = ["Random Forest", "XGBoost", "Gradient Boosting"]
importances_list = [
    importances_rf,
    importances_xgb,
    pd.DataFrame({"Feature": features, "Importance": gbr.feature_importances_}).sort_values("Importance", ascending=False)
]

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
for i, name in enumerate(tree_models):
    sns.barplot(data=importances_list[i], x="Importance", y="Feature", ax=axes4[i])
    axes4[i].set_title(f"{name} Feature Importances")

plt.tight_layout()
plt.savefig("feature_importances_tree_models.png", dpi=300)
plt.show()

joblib.dump(rf, "rsrp_rf_model.pkl")
joblib.dump(xgb_model, "rsrp_xgb_model.pkl")
joblib.dump(svr, "rsrp_svr_model.pkl")
joblib.dump(knn, "rsrp_knn_model.pkl")
joblib.dump(gbr, "rsrp_gbr_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels saved successfully.")
