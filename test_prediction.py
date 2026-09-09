import sys
sys.path.append('virtual-dcb')
from prediction_model import predictor

# Test with the exact input from the user's request
test_input = {
    'E11': 125.3,
    'E22': 8.4,
    'G12': 5.1,
    'poisson_ratio': 0.28,
    'ply_thickness': 25,
    'width': 22.54,
    'thickness': 3.3,
    'initial_crack_avg': 47.5,
    'loading_rate': 1.0
}

print("=" * 60)
print("TESTING PREDICTION PIPELINE")
print("=" * 60)
print("\nInput:")
for key, value in test_input.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("RUNNING PREDICTION")
print("=" * 60)

try:
    results = predictor.predict(test_input)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    force = results['Force_N']
    displacement = results['Displacement_mm']
    compliance = results['Compliance_mm_per_N']
    crack_length = results['Crack_Length_mm']
    serr = results['SERR_kJ_m2']
    
    print(f"Force: {force:.4f} N")
    print(f"Displacement: {displacement:.4f} mm")
    print(f"Compliance: {compliance:.6f} mm/N")
    print(f"Crack Length: {crack_length:.4f} mm")
    print(f"Strain Energy Release Rate: {serr:.4f} kJ/m²")
    
    print("\n" + "=" * 60)
    print("PHYSICAL CONSISTENCY CHECKS")
    print("=" * 60)
    
    # Check compliance consistency
    calc_compliance = displacement / force
    compliance_diff = abs(compliance - calc_compliance)
    print(f"Calculated Compliance (δ/P): {calc_compliance:.6f} mm/N")
    print(f"Predicted Compliance: {compliance:.6f} mm/N")
    print(f"Difference: {compliance_diff:.8f} mm/N")
    print(f"Consistency: {'✓ PASS' if compliance_diff < 0.001 else '✗ FAIL'}")
    
    # Check crack length consistency
    initial_crack = test_input['initial_crack_avg']
    crack_consistent = crack_length >= initial_crack
    print(f"\nInitial Crack Length: {initial_crack:.4f} mm")
    print(f"Final Crack Length: {crack_length:.4f} mm")
    print(f"Consistency: {'✓ PASS' if crack_consistent else '✗ FAIL'}")
    
    # Check graph data
    curve_data = results['curve_data']
    print(f"\nCurve data points: {len(curve_data['displacement'])}")
    print(f"Final displacement in curve: {curve_data['displacement'][-1]:.4f} mm")
    print(f"Final force in curve: {curve_data['force'][-1]:.4f} N")
    print(f"Final crack length in curve: {curve_data['crack_length'][-1]:.4f} mm")
    print(f"Graph endpoint matches results: {'✓ PASS' if abs(curve_data['displacement'][-1] - displacement) < 0.01 else '✗ FAIL'}")
    
    if results['warnings']:
        print("\n" + "=" * 60)
        print("WARNINGS")
        print("=" * 60)
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    else:
        print("\n" + "=" * 60)
        print("NO WARNINGS")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
