import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import time

print("=" * 70)
print("AI-Assisted Virtual DCB Testing Platform - Grouped Model Training")
print("=" * 70)

# 1. Load Dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'Final_DCB_Master_Dataset.csv')
print(f"\n1. Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"Raw dataset shape: {df.shape}")

# 2. Identify Unique Specimens
unique_specimens = df['Specimen_ID'].unique().tolist()
print(f"\nTotal Unique Specimens ({len(unique_specimens)}):")
for spec in unique_specimens:
    print(f" - {spec} ({df[df['Specimen_ID'] == spec]['Configuration'].iloc[0]})")

# 3. Clean Sensor Artifacts & Purge Optical Noise Before Crack Initiation
print("\n2. Cleaning sensor noise and enforcing pre-initiation physics...")
clean_mask = (
    (df['Force_N'] >= 0.5) &
    (df['Displacement_mm'] >= 0.5) &
    (df['Displacement_mm'] <= 45.0) &
    (df['Crack_Length_mm'] >= df['Initial_Crack_Avg_mm'] * 0.95) &
    (df['Crack_Length_mm'] <= 140.0)
)
df_clean = df[clean_mask].copy()

# In physical DCB tests (ASTM D5528), before delamination onset (peak load),
# the specimen bends elastically and the crack DOES NOT propagate (a = a0).
# Optical tracking camera noise prior to crack opening is purged:
for spec_id, g in df_clean.groupby('Specimen_ID'):
    p_max_idx = g['Force_N'].idxmax()
    delta_c = g.loc[p_max_idx, 'Displacement_mm']
    a0 = g['Initial_Crack_Avg_mm'].iloc[0]
    pre_mask = (df_clean['Specimen_ID'] == spec_id) & (df_clean['Displacement_mm'] <= delta_c)
    df_clean.loc[pre_mask, 'Crack_Length_mm'] = a0

print(f"Rows after filtering: {len(df_clean)} (from {len(df)})")

# 4. Zero-Leakage Grouped Train / Validation / Test Partitioning
# GroupShuffleSplit by Specimen_ID ensures zero row leakage across specimens
train_specs = ['DCB-BL-001', 'DCB-BL-002', 'DCB-BL-003', 'DCB-BL-004', 'DCB-RE-001', 'DCB-RE-002', 'DCB-RE-003', 'DCB-RE-004']
val_specs = ['DCB-BL-005', 'DCB-RE-005']
test_specs = ['DCB-BL-006', 'DCB-RE-006']

print("\n3. Partitioning by Specimen_ID (Zero Data Leakage):")
print(f"  Train Specimens ({len(train_specs)}): {train_specs}")
print(f"  Validation Specimens ({len(val_specs)}): {val_specs}")
print(f"  Test Specimens ({len(test_specs)}): {test_specs}")

input_features = [
    'E11_GPa', 'E22_GPa', 'G12_GPa', 'Poisson_Ratio', 'Ply_Thickness_um',
    'Loading_Rate_mm_min', 'Width_mm', 'Thickness_mm', 'Initial_Crack_Avg_mm',
    'Displacement_mm'
]
target_cols = ['Force_N', 'Crack_Length_mm']

train_df = df_clean[df_clean['Specimen_ID'].isin(train_specs)].copy()
val_df = df_clean[df_clean['Specimen_ID'].isin(val_specs)].copy()
test_df = df_clean[df_clean['Specimen_ID'].isin(test_specs)].copy()

# Subsample train for balanced representation
if len(train_df) > 60000:
    train_df = train_df.sample(n=60000, random_state=42).copy()

print(f"  Train Samples: {len(train_df)}, Val Samples: {len(val_df)}, Test Samples: {len(test_df)}")

# 5. Compute Training Domain Min/Max for All Inputs
dataset_stats = {}
for col in input_features:
    dataset_stats[col] = {
        'min': float(df_clean[col].min()),
        'max': float(df_clean[col].max()),
        'mean': float(df_clean[col].mean()),
        'std': float(df_clean[col].std())
    }

# 6. Fit StandardScaler Strictly on Training Set
scaler = StandardScaler()
X_train = train_df[input_features].values
y_train = train_df[target_cols].values
X_train_scaled = scaler.fit_transform(X_train)

X_test = test_df[input_features].values
y_test = test_df[target_cols].values
X_test_scaled = scaler.transform(X_test)

# 7. Train ExtraTreesRegressor
print("\n4. Training Multi-Output ExtraTreesRegressor...")
t0 = time.time()
model = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
train_time = time.time() - t0
print(f"Model training completed in {train_time:.2f} seconds!")

# 8. Evaluate on Completely Unseen Test Specimens (DCB-BL-006 and DCB-RE-006)
print(f"\n5. Evaluating Generalization on Unseen Test Specimens {test_specs}...")
y_pred = model.predict(X_test_scaled)

force_test = y_test[:, 0]
force_pred = y_pred[:, 0]
crack_test = y_test[:, 1]
crack_pred = y_pred[:, 1]

force_r2 = float(r2_score(force_test, force_pred))
force_mae = float(mean_absolute_error(force_test, force_pred))
force_rmse = float(np.sqrt(mean_squared_error(force_test, force_pred)))

crack_r2 = float(r2_score(crack_test, crack_pred))
crack_mae = float(mean_absolute_error(crack_test, crack_pred))
crack_rmse = float(np.sqrt(mean_squared_error(crack_test, crack_pred)))

print(f"  [Force_N]         R2: {force_r2:.4f} | MAE: {force_mae:.3f} N  | RMSE: {force_rmse:.3f} N")
print(f"  [Crack_Length_mm] R2: {crack_r2:.4f} | MAE: {crack_mae:.3f} mm | RMSE: {crack_rmse:.3f} mm")

metrics_summary = {
    'model_name': 'ExtraTreesRegressor (Grouped Specimen DCB Model)',
    'grouping_column': 'Specimen_ID',
    'unique_specimens_total': len(unique_specimens),
    'train_specimens': train_specs,
    'val_specimens': val_specs,
    'test_specimens': test_specs,
    'training_samples': int(len(train_df)),
    'test_samples': int(len(test_df)),
    'training_time_seconds': round(train_time, 2),
    'features': input_features,
    'targets': target_cols,
    'test_evaluation_unseen_specimens': {
        'force': {
            'r2': round(force_r2, 4),
            'mae_N': round(force_mae, 4),
            'rmse_N': round(force_rmse, 4)
        },
        'crack_length': {
            'r2': round(crack_r2, 4),
            'mae_mm': round(crack_mae, 4),
            'rmse_mm': round(crack_rmse, 4)
        }
    }
}

# 9. Save Artifacts
print("\n6. Saving Model and Metadata Artifacts...")
model_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'trained_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
feature_cols_path = os.path.join(model_dir, 'feature_cols.pkl')
target_cols_path = os.path.join(model_dir, 'target_cols.pkl')
stats_path = os.path.join(model_dir, 'dataset_stats.json')
metrics_path = os.path.join(model_dir, 'model_metrics.json')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(input_features, feature_cols_path)
joblib.dump(target_cols, target_cols_path)

with open(stats_path, 'w') as f:
    json.dump(dataset_stats, f, indent=4)

with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=4)

print(f"[OK] Model saved to: {model_path}")
print(f"[OK] Scaler saved to: {scaler_path}")
print(f"[OK] Features saved to: {feature_cols_path}")
print(f"[OK] Targets saved to: {target_cols_path}")
print(f"[OK] Dataset stats saved to: {stats_path}")
print(f"[OK] Model metrics saved to: {metrics_path}")

print("\n" + "=" * 70)
print("ZERO-LEAKAGE GROUPED TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
