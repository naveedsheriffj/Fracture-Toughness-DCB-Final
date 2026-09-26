# Fracture-Toughness-DCB: AI-Assisted Virtual DCB Testing Platform

A scientific, physics-informed machine learning platform for simulating and analyzing Mode-I delamination, fracture behavior, and R-curves of Carbon Fiber Reinforced Polymer (CFRP) composite laminates tested under the **ASTM D5528** Double Cantilever Beam (DCB) standard.

Designed to run natively in **Python IDLE** (or terminal) with interactive command menus and multi-panel Matplotlib fracture mechanics plots.

---

## Key Capabilities & Scientific Architecture

1. **Zero Data Leakage Machine Learning Pipeline**:
   - Model trained using `Specimen_ID` grouped partitioning (GroupSplit), completely isolating unseen test specimens from training and validation sets.
   - High-performance `ExtraTreesRegressor` ensemble predicting coupled crosshead load ($P$) and delamination crack extension ($a$) along controlled displacement trajectories ($\delta = 1.0 - 35.0$ mm).

2. **Strict Physics Consistency Guarantees**:
   - **Initial Crack Length Consistency ($a[0] = a_0$)**: The initial crack length starts strictly at the user-specified initial crack length $a_0$ without experimental noise offset.
   - **Pre-Initiation Boundary Condition**: Delamination does not advance before critical crack initiation; for all $\delta \le \delta_{\text{crit}}$, crack length is strictly maintained at $a(\delta) = a_0$.
   - **Monotonic Delamination Growth**: Enforces irreversible crack propagation ($\frac{da}{d\delta} \ge 0$).
   - **Compliance Law ($C = \delta / P$)**: Specimen compliance satisfies $C(\delta) = \delta / P(\delta)$ across all 70 discrete points with zero mathematical drift.
   - **ASTM D5528 Modified Beam Theory (MBT)**: Mode-I Strain Energy Release Rate ($G_I$) calculated analytically using standard MBT formulation with beam rotation correction ($|\Delta| = 2.5$ mm):
     $$G_I = \frac{3 P \delta}{2 b (a + |\Delta|)}$$
   - **Single Authoritative Data Source**: All printed tabular outputs and all 5 interactive Matplotlib plots consume the exact same underlying 70-point prediction curve.
   - **Out-of-Distribution (OOD) Domain Evaluation**: Validates user inputs against experimental training distribution bounds and raises advisory warnings for extrapolated parameters.

3. **Multi-Panel Matplotlib Visualizer**:
   - Generates 5 ASTM D5528 analysis curves simultaneously:
     1. **Load vs. Displacement ($P - \delta$)** with peak initiation load annotated
     2. **Crack Length vs. Displacement ($a - \delta$)** showing pre-initiation plateau
     3. **Load vs. Crack Length ($P - a$)**
     4. **Compliance vs. Crack Length ($C - a$)**
     5. **Mode-I Resistance Curve / R-Curve ($G_I - a$)** with $G_{\text{Ic}}$ annotated
     6. **Fracture Summary Box** with critical values and physics verification checklist

---

## Model Validation Metrics

Evaluated across unseen test specimens grouped by `Specimen_ID` with zero data leakage:

| Target Variable | Physical Unit | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: |
| **Applied Force ($P$)** | $\text{N}$ | **0.81** | **3.36 N** | **4.09 N** |
| **Displacement ($\delta$)** | $\text{mm}$ | **0.99** | **0.59 mm** | **0.86 mm** |
| **Crack Length ($a$)** | $\text{mm}$ | **0.95** | **4.47 mm** | **5.90 mm** |
| **Strain Energy Release Rate ($G_I$)** | $\text{kJ/m}^2$ | **0.53** | **0.05 kJ/m²** | **0.07 kJ/m²** |

---

## Project Structure

```
Fracture-Toughness-DCB/
│
├── Final_DCB_Master_Dataset.csv    # Experimental master dataset
├── run_dcb.py                      # Main interactive runner script for Python IDLE
├── main.py                         # Clean entrypoint (calls run_dcb.py)
├── prediction_model.py             # Physics-informed prediction engine (DCBPredictor)
├── train_model.py                  # Zero-leakage ML training & validation pipeline
├── test_prediction.py              # Standalone prediction & consistency test harness
├── analyze_dataset.py              # Dataset distribution & statistical checks
├── check_crack_data.py             # Crack propagation validation script
├── check_force_data.py             # Load curve validation script
├── requirements.txt                # Python dependencies (NumPy, SciPy, scikit-learn, pandas, Matplotlib)
│
└── model/                          # Serialized ML artifacts
    ├── trained_model.pkl           # ExtraTreesRegressor model binary
    ├── scaler.pkl                  # Fitted StandardScaler feature scaler
    ├── feature_cols.pkl            # Ordered input feature column names
    ├── target_cols.pkl             # Target output column names [Force_N, Crack_Length_mm]
    ├── dataset_stats.json          # Experimental training domain min/max bounds
    └── model_metrics.json          # Cross-specimen validation metrics
```

---

## Quick Start: Running in Python IDLE

### 1. Install Dependencies
In your terminal or Command Prompt, run:
```bash
pip install -r requirements.txt
```

### 2. Run in Python IDLE
1. Open **Python IDLE**.
2. Go to **File -> Open...** and select `run_dcb.py` (or `main.py`).
3. Press **`F5`** (or select **Run -> Run Module**).
4. The interactive console menu will appear in the Python IDLE Shell:
   ```text
   Select an action:
     [1] Run Baseline CFRP Specimen (Nominal: E11=125.3 GPa, a0=47.5 mm)
     [2] Run High-Modulus Specimen [OOD Demo: E11=200.0 GPa]
     [3] Run Deep Notch Specimen [OOD Demo: a0=60.0 mm]
     [4] Enter Custom Specimen Properties
     [5] Show ML Model Metrics & Validation Summary
     [6] Retrain Model from Master Dataset
     [7] Exit
   ```
5. Entering `1`, `2`, `3`, or `4` will:
   - Calculate critical initiation parameters ($P_{\text{crit}}, \delta_{\text{crit}}, C_{\text{crit}}, a_0, G_{\text{Ic}}$).
   - Calculate test termination parameters ($P_{\text{final}}, \delta_{\text{final}}, C_{\text{final}}, a_{\text{final}}, G_{I,\text{final}}$).
   - Perform automated scientific consistency audits ($a[0] = a_0$, $C = \delta / P$).
   - Pop up the interactive **Matplotlib** window showing all 5 fracture curves.
   - Closing the plot window returns you to the menu to run another simulation.

---

## License & Disclaimer

AI model predictions are intended for research, surrogate modeling, and virtual testing analysis. Critical aerospace and structural composite designs must be verified with physical DCB testing according to ASTM D5528 protocols.
