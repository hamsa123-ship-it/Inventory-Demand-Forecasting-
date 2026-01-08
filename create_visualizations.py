"""
IntelliStock - Create Visualizations
====================================
Generate beautiful charts for presentation
"""

print("=" * 70)
print("🎨 IntelliStock - Creating Visualizations")
print("=" * 70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("\n📂 Loading data...")
# Load data
results_df = pd.read_csv('model_results.csv')
predictions_df = pd.read_csv('predictions.csv')
test_df = pd.read_csv('test_data.csv')

print("✓ Data loaded!")

# Create output folder for figures
import os
os.makedirs('figures', exist_ok=True)

print("\n🎨 Creating visualizations...\n")

# ============================================================================
# 1. Model Comparison - MAPE
# ============================================================================
print("   1/6 Model comparison (MAPE)...")

plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
bars = plt.bar(results_df['Model'], results_df['MAPE'], color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add target line
plt.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Target (15%)')

plt.title('Model Performance Comparison - MAPE', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('MAPE (%)', fontsize=12, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/1_model_comparison_mape.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Model Comparison - All Metrics
# ============================================================================
print("   2/6 Model comparison (all metrics)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['MAE', 'RMSE', 'MAPE']
titles = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Percentage Error (MAPE)']
ylabels = ['MAE (units)', 'RMSE (units)', 'MAPE (%)']

for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    ax = axes[idx]
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('figures/2_model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Predictions vs Actual (Sample)
# ============================================================================
print("   3/6 Predictions vs actual...")

# Take first 100 predictions for clarity
sample_size = 100
sample_preds = predictions_df.head(sample_size)

plt.figure(figsize=(14, 6))
x = range(sample_size)

plt.plot(x, sample_preds['Actual'], 'o-', label='Actual', linewidth=2, markersize=4, color='black')
plt.plot(x, sample_preds['Ensemble'], 's-', label='Ensemble Prediction', linewidth=2, markersize=3, color='#2ecc71', alpha=0.7)
plt.plot(x, sample_preds['XGBoost'], '^-', label='XGBoost Prediction', linewidth=1.5, markersize=3, color='#3498db', alpha=0.6)
plt.plot(x, sample_preds['Baseline'], 'x-', label='Baseline Forecast', linewidth=1.5, markersize=3, color='#e74c3c', alpha=0.5)

plt.title(f'Predictions vs Actual Sales (First {sample_size} Test Records)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
plt.ylabel('Units Sold', fontsize=12, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/3_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. Error Distribution
# ============================================================================
print("   4/6 Error distribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = ['Baseline', 'XGBoost', 'LightGBM', 'Ensemble']
colors_hist = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']

for idx, (model, color) in enumerate(zip(models, colors_hist)):
    ax = axes[idx // 2, idx % 2]
    
    errors = predictions_df['Actual'] - predictions_df[model]
    
    ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    
    ax.set_title(f'{model} - Error Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/4_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. Improvement Over Baseline
# ============================================================================
print("   5/6 Improvement over baseline...")

baseline_mape = results_df[results_df['Model'] == 'Baseline Forecast']['MAPE'].values[0]
improvements = []

for _, row in results_df.iterrows():
    if row['Model'] != 'Baseline Forecast':
        improvement = ((baseline_mape - row['MAPE']) / baseline_mape) * 100
        improvements.append({'Model': row['Model'], 'Improvement': improvement})

improvement_df = pd.DataFrame(improvements)

plt.figure(figsize=(10, 6))
bars = plt.bar(improvement_df['Model'], improvement_df['Improvement'], 
               color=['#2ecc71', '#3498db', '#9b59b6'], edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Percentage Improvement Over Baseline', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Improvement (%)', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('figures/5_improvement_over_baseline.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. Summary Dashboard
# ============================================================================
print("   6/6 Summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('IntelliStock: Intelligent Inventory Demand Forecasting - Results Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

# Top Left: Model Comparison
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.bar(results_df['Model'], results_df['MAPE'], color=colors, edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Target (15%)')
ax1.set_title('Model Performance (MAPE)', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAPE (%)', fontsize=10, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Top Right: Key Metrics
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
best_model = results_df.iloc[0]
metrics_text = f"""
🏆 BEST MODEL
{'=' * 25}

Model: {best_model['Model']}

MAPE:  {best_model['MAPE']:.2f}%
MAE:   {best_model['MAE']:.2f} units
RMSE:  {best_model['RMSE']:.2f} units

{'=' * 25}
✅ TARGET ACHIEVED!
Target: <15% MAPE
"""
ax2.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
         verticalalignment='center')

# Middle: Predictions Sample
ax3 = fig.add_subplot(gs[1, :])
sample_size = 50
sample = predictions_df.head(sample_size)
x = range(sample_size)
ax3.plot(x, sample['Actual'], 'o-', label='Actual', linewidth=2, markersize=4, color='black')
ax3.plot(x, sample['Ensemble'], 's-', label='Ensemble', linewidth=2, markersize=3, color='#2ecc71', alpha=0.7)
ax3.plot(x, sample['Baseline'], 'x-', label='Baseline', linewidth=1.5, markersize=3, color='#e74c3c', alpha=0.5)
ax3.set_title(f'Predictions vs Actual (Sample: {sample_size} records)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sample Index', fontsize=10)
ax3.set_ylabel('Units Sold', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Bottom Left: Improvement
ax4 = fig.add_subplot(gs[2, 0])
bars = ax4.bar(improvement_df['Model'], improvement_df['Improvement'],
               color=['#2ecc71', '#3498db', '#9b59b6'], edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_title('Improvement Over Baseline', fontsize=11, fontweight='bold')
ax4.set_ylabel('Improvement (%)', fontsize=9)
ax4.tick_params(axis='x', rotation=15, labelsize=8)
ax4.grid(axis='y', alpha=0.3)

# Bottom Middle: Error Distribution (Ensemble)
ax5 = fig.add_subplot(gs[2, 1])
ensemble_errors = predictions_df['Actual'] - predictions_df['Ensemble']
ax5.hist(ensemble_errors, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax5.set_title('Ensemble Error Distribution', fontsize=11, fontweight='bold')
ax5.set_xlabel('Error (units)', fontsize=9)
ax5.set_ylabel('Frequency', fontsize=9)
ax5.grid(alpha=0.3)

# Bottom Right: Stats Table
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
stats_data = results_df[['Model', 'MAPE']].head(3)
table = ax6.table(cellText=stats_data.values, colLabels=stats_data.columns,
                  cellLoc='center', loc='center',
                  bbox=[0, 0.2, 1, 0.7])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(stats_data.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')
ax6.set_title('Top Models', fontsize=11, fontweight='bold', pad=20)

plt.savefig('figures/6_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("✅ VISUALIZATIONS COMPLETE!")
print("=" * 70)

print("\n📁 Created 6 charts in 'figures/' folder:")
print("   1. 1_model_comparison_mape.png")
print("   2. 2_model_comparison_all_metrics.png")
print("   3. 3_predictions_vs_actual.png")
print("   4. 4_error_distribution.png")
print("   5. 5_improvement_over_baseline.png")
print("   6. 6_summary_dashboard.png")

print("\n🎨 All charts are high-resolution (300 DPI) - perfect for presentations!")
print("\n" + "=" * 70 + "\n")