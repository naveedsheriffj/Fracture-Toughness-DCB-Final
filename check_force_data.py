import pandas as pd

df = pd.read_csv('Final_DCB_Master_Dataset.csv')

print("=" * 60)
print("FORCE DATA ANALYSIS")
print("=" * 60)

baseline = df[df['Configuration'] == 'Baseline']
print(f"\nBaseline data rows: {len(baseline)}")

print("\nForce Statistics:")
print(baseline['Force_N'].describe())

print("\nDisplacement Statistics:")
print(baseline['Displacement_mm'].describe())

print("\nSample of force vs displacement:")
sample = baseline[['Displacement_mm', 'Force_N']].head(20)
print(sample)

print("\nCompliance Statistics:")
print(baseline['Compliance_mm_per_N'].describe())
