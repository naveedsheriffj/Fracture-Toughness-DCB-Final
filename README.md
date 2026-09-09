# Fracture-Toughness-DCB: AI-Assisted Virtual DCB Testing Platform

A scientific, physics-informed machine learning web platform for simulating and analyzing Mode-I delamination, fracture behavior, and R-curves of Carbon Fiber Reinforced Polymer (CFRP) composite laminates tested under the **ASTM D5528** Double Cantilever Beam (DCB) standard.

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
   - **Single Authoritative Data Source**: All scalar result cards and all 5 interactive charts consume the exact same underlying 70-point prediction curve.
   - **Out-of-Distribution (OOD) Domain Evaluation**: Validates user inputs against experimental training distribution bounds and raises advisory warnings for extrapolated parameters.

3. **Standard Scientific Notation & Units**:
   - All input labels, result cards, and graphs follow international composite fracture mechanics standards with full LaTeX math typesetting via **MathJax 3**.
   - Individual one-by-one prediction display cards detailing: *"For the given material property, the predicted [parameter] is [value]"*.
   - Each analysis chart features a dedicated right-side tab displaying the **Peak Value** and **Low Value** along with coordinate locations.

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
├── analyze_dataset.py              # Dataset distribution & statistical checks
├── check_crack_data.py             # Crack propagation validation script
├── check_force_data.py             # Load curve validation script
├── test_prediction.py              # Standalone prediction test harness
│
└── virtual-dcb/                    # Full-stack web application
    ├── app.py                      # Flask backend application & REST API
    ├── train_model.py              # Zero-leakage ML training & validation pipeline
    ├── prediction_model.py         # Physics-informed prediction engine (DCBPredictor)
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Platform documentation
    │
    ├── dataset/
    │   └── Final_DCB_Master_Dataset.csv
    │
    ├── model/
    │   ├── trained_model.pkl       # Serialized ExtraTreesRegressor model (Git LFS)
    │   ├── scaler.pkl              # Fitted StandardScaler feature scaler
    │   ├── feature_cols.pkl        # Ordered input feature column names
    │   ├── target_cols.pkl         # Target output column names [Force, Crack]
    │   ├── dataset_stats.json      # Experimental training domain min/max bounds
    │   └── model_metrics.json      # Cross-specimen validation metrics
    │
    ├── templates/
    │   └── index.html              # Frontend UI with MathJax 3 & side stats tabs
    │
    └── static/
        ├── style.css               # Responsive styling
        └── script.js               # Frontend controller, Chart.js lifecycle, extremes tabs
```

---

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/naveedsheriffj/Fracture-Toughness-DCB.git
cd Fracture-Toughness-DCB/virtual-dcb

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application
```bash
python app.py
```
Open your browser at **`http://localhost:5000`**.

---

## License & Disclaimer

AI model predictions are intended for research, surrogate modeling, and virtual testing analysis. Critical aerospace and structural composite designs must be verified with physical DCB testing according to ASTM D5528 protocols.
