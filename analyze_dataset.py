import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Final_DCB_Master_Dataset.csv')

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("COLUMN STATISTICS")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("UNIQUE SPECIMENS")
print("=" * 60)
print(f"Unique specimens: {df['Specimen_ID'].nunique()}")

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)
print(df.head(10))

print("\n" + "=" * 60)
print("COMPLIANCE CONSISTENCY CHECK")
print("=" * 60)
# Check if Compliance = Displacement / Force
df['calc_compliance'] = df['Displacement_mm'] / df['Force_N']
df['compliance_diff'] = abs(df['Compliance_mm_per_N'] - df['calc_compliance'])
print(f"Mean absolute difference: {df['compliance_diff'].mean()}")
print(f"Max absolute difference: {df['compliance_diff'].max()}")
print(f"Percentage with diff > 0.01: {(df['compliance_diff'] > 0.01).sum() / len(df) * 100:.2f}%")

print("\n" + "=" * 60)
print("CRACK LENGTH CONSISTENCY CHECK")
print("=" * 60)
# Check if crack length is always >= initial crack
df['crack_consistent'] = df['Crack_Length_mm'] >= df['Initial_Crack_Avg_mm']
print(f"Rows where crack length < initial crack: {(~df['crack_consistent']).sum()}")
print(f"Percentage: {(~df['crack_consistent']).sum() / len(df) * 100:.2f}%")

print("\n" + "=" * 60)
print("NEGATIVE VALUES CHECK")
print("=" * 60)
print("Negative Force_N:", (df['Force_N'] < 0).sum())
print("Negative Displacement_mm:", (df['Displacement_mm'] < 0).sum())
print("Negative Compliance_mm_per_N:", (df['Compliance_mm_per_N'] < 0).sum())
print("Negative SERR_kJ_m2:", (df['SERR_kJ_m2'] < 0).sum())
