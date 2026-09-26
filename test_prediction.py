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
    print("PREDICTION RESULTS (TERMINATION STATE)")
    print("=" * 60)
    
    preds = results['predictions']
    crit = results['critical_point']
    curve = results['prediction_curve']
    
    print(f"Force: {preds['force']}")
    print(f"Displacement: {preds['displacement']}")
    print(f"Compliance: {preds['compliance']}")
    print(f"Crack Length: {preds['crack_length']}")
    print(f"Strain Energy Release Rate: {preds['serr']}")
    
    print("\n" + "=" * 60)
    print("CRITICAL INITIATION POINT (PEAK LOAD)")
    print("=" * 60)
    print(f"Critical Force: {crit['critical_force']}")
    print(f"Critical Displacement: {crit['critical_displacement']}")
    print(f"Critical Compliance: {crit['critical_compliance']}")
    print(f"Critical Crack Length: {crit['critical_crack']}")
    print(f"Initiation G_Ic: {crit['initiation_Gic']}")
    
    print("\n" + "=" * 60)
    print("PHYSICAL CONSISTENCY CHECKS")
    print("=" * 60)
    
    # Check compliance consistency on final point
    final_pt = curve[-1]
    calc_compliance = final_pt['displacement'] / final_pt['load']
    compliance_diff = abs(final_pt['compliance'] - calc_compliance)
    print(f"Calculated Compliance (delta/P): {calc_compliance:.6f} mm/N")
    print(f"Predicted Compliance: {final_pt['compliance']:.6f} mm/N")
    print(f"Difference: {compliance_diff:.8f} mm/N")
    print(f"Consistency: {'[PASS]' if compliance_diff < 0.001 else '[FAIL]'}")
    
    # Check crack length consistency
    initial_crack = test_input['initial_crack_avg']
    crack_consistent = final_pt['crackLength'] >= initial_crack
    print(f"\nInitial Crack Length: {initial_crack:.4f} mm")
    print(f"Final Crack Length: {final_pt['crackLength']:.4f} mm")
    print(f"Consistency: {'[PASS]' if crack_consistent else '[FAIL]'}")
    
    # Check graph data
    print(f"\nCurve data points: {len(curve)}")
    print(f"Initial crack in curve: {curve[0]['crackLength']:.4f} mm")
    print(f"Final displacement in curve: {curve[-1]['displacement']:.4f} mm")
    print(f"Final force in curve: {curve[-1]['load']:.4f} N")
    print(f"Final crack length in curve: {curve[-1]['crackLength']:.4f} mm")
    print(f"Initial crack starts at a0: {'[PASS]' if curve[0]['crackLength'] == initial_crack else '[FAIL]'}")
    
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
