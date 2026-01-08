"""
IntelliStock - Baseline Models
==============================
Train and compare multiple forecasting models
"""

print("=" * 70)
print("🚀 IntelliStock - Phase 2: Baseline Models")
print("=" * 70)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load processed data
print("\n📂 Loading data...")
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

print(f"✓ Train: {len(train_df):,} records")
print(f"✓ Test: {len(test_df):,} records")

# Prepare features and target
print("\n🔧 Preparing features...")

# Select relevant features (exclude date, IDs, and target)
exclude_cols = ['Date', 'Store ID', 'Product ID', 'Units Sold', 'Demand Forecast', 'Category', 'Region', 'Weather Condition', 'Seasonality']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df['Units Sold']
X_test = test_df[feature_cols]
y_test = test_df['Units Sold']

print(f"✓ Using {len(feature_cols)} features")
print(f"✓ Target: Units Sold")

# Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    """Calculate performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    print(f"\n📊 {model_name} Results:")
    print(f"   MAE:  {mae:.2f} units")
    print(f"   RMSE: {rmse:.2f} units")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Store results
results = []

print("\n" + "=" * 70)
print("MODEL 1: Baseline (Current Forecast)")
print("=" * 70)

y_baseline = test_df['Demand Forecast']
baseline_results = calculate_metrics(y_test, y_baseline, 'Baseline Forecast')
results.append(baseline_results)

print("\n" + "=" * 70)
print("MODEL 2: XGBoost")
print("=" * 70)
print("Training XGBoost... (this takes ~30 seconds)")

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train, y_train)
print("✓ Training complete!")

y_pred_xgb = xgb_model.predict(X_test)
xgb_results = calculate_metrics(y_test, y_pred_xgb, 'XGBoost')
results.append(xgb_results)

print("\n" + "=" * 70)
print("MODEL 3: LightGBM")
print("=" * 70)
print("Training LightGBM... (this takes ~20 seconds)")

from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

lgbm_model.fit(X_train, y_train)
print("✓ Training complete!")

y_pred_lgbm = lgbm_model.predict(X_test)
lgbm_results = calculate_metrics(y_test, y_pred_lgbm, 'LightGBM')
results.append(lgbm_results)

print("\n" + "=" * 70)
print("MODEL 4: Simple Ensemble (XGBoost + LightGBM)")
print("=" * 70)
print("Creating ensemble...")

# Average predictions from both models
y_pred_ensemble = (y_pred_xgb + y_pred_lgbm) / 2
ensemble_results = calculate_metrics(y_test, y_pred_ensemble, 'Ensemble')
results.append(ensemble_results)

# Results comparison
print("\n" + "=" * 70)
print("📊 FINAL RESULTS COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAPE')

print("\n" + results_df.to_string(index=False))

# Find best model
best_model = results_df.iloc[0]
print("\n" + "=" * 70)
print("🏆 BEST MODEL")
print("=" * 70)
print(f"\n🥇 {best_model['Model']}")
print(f"   MAPE: {best_model['MAPE']:.2f}%")
print(f"   MAE:  {best_model['MAE']:.2f} units")
print(f"   RMSE: {best_model['RMSE']:.2f} units")

# Check if we beat the target
print("\n" + "=" * 70)
print("🎯 TARGET ACHIEVEMENT")
print("=" * 70)
print(f"\n   Baseline MAPE:     {baseline_results['MAPE']:.2f}%")
print(f"   Best Model MAPE:   {best_model['MAPE']:.2f}%")
print(f"   Target MAPE:       <15.00%")

if best_model['MAPE'] < 15:
    print("\n   🎉 TARGET ACHIEVED! ✅")
else:
    print(f"\n   📈 Progress: {baseline_results['MAPE'] - best_model['MAPE']:.2f}% improvement")
    print(f"   📉 Still need: {best_model['MAPE'] - 15:.2f}% to reach target")

# Save results
print("\n" + "=" * 70)
print("💾 Saving results...")
print("=" * 70)

results_df.to_csv('model_results.csv', index=False)
print("✓ Saved model_results.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Baseline': y_baseline.values,
    'XGBoost': y_pred_xgb,
    'LightGBM': y_pred_lgbm,
    'Ensemble': y_pred_ensemble
})
predictions_df.to_csv('predictions.csv', index=False)
print("✓ Saved predictions.csv")

print("\n" + "=" * 70)
print("✅ PHASE 2 COMPLETE!")
print("=" * 70)
print("\n🎯 Next Steps:")
print("   1. Check model_results.csv for detailed comparison")
print("   2. Check predictions.csv for individual predictions")
print("   3. Ready for Phase 3: Deep Learning models (LSTM/GRU)")
print("\n" + "=" * 70 + "\n")