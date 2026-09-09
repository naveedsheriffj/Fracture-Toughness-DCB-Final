import pandas as pd

df = pd.read_csv('Final_DCB_Master_Dataset.csv')

print("=" * 60)
print("CRACK LENGTH ANALYSIS")
print("=" * 60)

baseline = df[df['Configuration'] == 'Baseline']
print(f"\nBaseline data rows: {len(baseline)}")

print("\nCrack Length Statistics:")
print(baseline['Crack_Length_mm'].describe())

print("\nInitial Crack Average Statistics:")
print(baseline['Initial_Crack_Avg_mm'].describe())

print("\nSample of crack lengths vs initial cracks:")
sample = baseline[['Initial_Crack_Avg_mm', 'Crack_Length_mm', 'Displacement_mm']].head(20)
print(sample)

print("\nCrack growth (final - initial):")
crack_growth = baseline['Crack_Length_mm'] - baseline['Initial_Crack_Avg_mm']
print(crack_growth.describe())

print("\nPercentage where crack < initial crack:")
print((crack_growth < 0).sum() / len(baseline) * 100, "%")
