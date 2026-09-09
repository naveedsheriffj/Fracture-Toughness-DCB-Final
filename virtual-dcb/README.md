# AI-Assisted Virtual DCB Testing Platform

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
   - **Single Authoritative Data Source**: All scalar result cards and all 4 interactive charts consume the exact same underlying 70-point prediction curve.
   - **Out-of-Distribution (OOD) Domain Evaluation**: Validates user inputs against experimental training distribution bounds and raises advisory warnings for extrapolated parameters.

3. **Standard Scientific Notation & Units**:
   - All input labels, result cards, and graphs follow international composite fracture mechanics standards with full LaTeX math typesetting via **MathJax 3**.

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

## Input Parameters & Scientific Units

### 1. Specimen Dimensions
| Parameter Label in UI | Symbol | Physical Quantity | Standard Unit | Typical Range |
| :--- | :---: | :--- | :---: | :---: |
| `Width \(b\) (mm)` | $b$ | Specimen arm width | $\text{mm}$ | $20.0 - 25.0$ |
| `Thickness \(2h\) (mm)` | $2h$ | Total laminate thickness (both arms) | $\text{mm}$ | $3.0 - 5.0$ |
| `Initial Crack Length \(a_0\) (mm)` | $a_0$ | Initial delamination starter crack | $\text{mm}$ | $45.0 - 65.0$ |

### 2. Material Properties
| Parameter Label in UI | Symbol | Physical Quantity | Standard Unit | Baseline CFRP |
| :--- | :---: | :--- | :---: | :---: |
| `Longitudinal Modulus \(E_{11}\) (GPa)` | $E_{11}$ | Axial Young's modulus | $\text{GPa}$ | $125.3$ |
| `Transverse Modulus \(E_{22}\) (GPa)` | $E_{22}$ | Transverse Young's modulus | $\text{GPa}$ | $8.4$ |
| `Shear Modulus \(G_{12}\) (GPa)` | $G_{12}$ | In-plane shear modulus | $\text{GPa}$ | $5.1$ |
| `Poisson's Ratio \(\nu_{12}\)` | $\nu_{12}$ | Major in-plane Poisson's ratio | Dimensionless | $0.28$ |
| `Ply Thickness \(t_{ply}\) (µm)` | $t_{\text{ply}}$ | Single ply cured thickness | $\mu\text{m}$ | $25.0$ |

### 3. Loading Conditions
| Parameter Label in UI | Symbol | Physical Quantity | Standard Unit | Standard Rate |
| :--- | :---: | :--- | :---: | :---: |
| `Loading Rate \(\dot{\delta}\) (mm/min)` | $\dot{\delta}$ | Crosshead displacement rate | $\text{mm/min}$ | $1.0$ |

---

## Output Parameters & Generated Curves

### Predicted State Outputs
- **Delamination Initiation Point (Peak Load State)**:
  - Critical Load: $P_{\text{crit}}$ ($\text{N}$)
  - Critical Displacement: $\delta_{\text{crit}}$ ($\text{mm}$)
  - Critical Compliance: $C_{\text{crit}}$ ($\text{mm/N}$)
  - Critical Crack Length: $a_0$ ($\text{mm}$)
  - Mode-I Initiation Toughness: $G_{Ic}$ ($\text{kJ/m}^2$)
- **Final Termination State ($\delta = 35.0$ mm)**:
  - Final Load: $P$ ($\text{N}$)
  - Final Displacement: $\delta$ ($\text{mm}$)
  - Final Compliance: $C = \delta / P$ ($\text{mm/N}$)
  - Final Crack Length: $a$ ($\text{mm}$)
  - Final SERR: $G_I$ ($\text{kJ/m}^2$)

### Interactive Analysis Graphs (Chart.js)
The platform displays 5 sequential Mode-I fracture analysis graphs arranged one after another:
1. **Load vs Displacement ($P - \delta$)**: Pre-initiation linear-elastic loading followed by delamination softening.
2. **Crack Length vs Displacement ($a - \delta$)**: Strict horizontal plateau at starter crack $a_0$ up to $\delta_{\text{crit}}$, transitioning to monotonic crack propagation.
3. **Load vs Crack Length ($P - a$)**: Delamination driving force evolution as crack advances through the laminate.
4. **Compliance vs Crack Length ($C - a$)**: Increasing structural flexibility ($C = \delta / P$) as delamination extends along the cantilever arms.
5. **GIC vs Crack Length (R - Curve, $G_{IC} - a$)**: Delamination resistance curve showing fracture energy release rate vs. crack propagation.

---

## Project Structure

```
virtual-dcb/
│
├── app.py                      # Flask backend application & REST API
├── train_model.py              # Zero-leakage ML training & validation pipeline
├── prediction_model.py         # Physics-informed prediction engine (DCBPredictor)
├── requirements.txt            # Python environment dependencies
├── README.md                   # Platform documentation & scientific guide
│
├── dataset/
│   └── Final_DCB_Master_Dataset.csv   # DCB experimental master dataset
│
├── model/
│   ├── trained_model.pkl       # Serialized ExtraTreesRegressor model
│   ├── scaler.pkl              # Fitted StandardScaler feature scaler
│   ├── feature_cols.pkl        # Ordered input feature column names
│   ├── target_cols.pkl         # Target output column names [Force, Crack]
│   ├── dataset_stats.json      # Experimental training domain min/max bounds
│   └── model_metrics.json      # Zero-leakage cross-specimen validation metrics
│
├── templates/
│   └── index.html              # Frontend UI with MathJax 3 & validation panel
│
└── static/
    ├── style.css               # Modern responsive CSS styling
    └── script.js               # Frontend controller, Chart.js lifecycle, consistency checks
```

---

## Prerequisites & Installation

### 1. Requirements
- **Python 3.8 - 3.12**
- Git (optional)

### 2. Environment Setup
Clone or navigate to the repository directory:
```bash
cd virtual-dcb
```

Create and activate a virtual environment:
- **Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

Install required dependencies:
```bash
pip install -r requirements.txt
```

---

## Training the Model

The platform includes pre-trained artifacts in the `model/` directory. To retrain the model from the master dataset:

```bash
python train_model.py
```

The script will:
1. Load `dataset/Final_DCB_Master_Dataset.csv`.
2. Clean and filter valid mechanical test data ($\delta \ge 1.0$ mm, $P > 0$ N, $a \ge a_0$).
3. Partition specimens by `Specimen_ID` into train, validation, and test subsets.
4. Train an `ExtraTreesRegressor` on 10 standardized physical features.
5. Compute and export performance metrics (`model_metrics.json`) and parameter bounds (`dataset_stats.json`).

---

## Running the Web Application

Start the Flask server:
```bash
python app.py
```

Access the interactive dashboard in your web browser:
```
http://localhost:5000
```

---

## Preset Configurations

The UI includes one-click preset buttons for rapid testing and out-of-distribution demonstration:

1. **Load Baseline Specimen**:
   - $E_{11} = 125.3$ GPa, $b = 22.54$ mm, $2h = 3.3$ mm, $a_0 = 47.5$ mm
   - Within nominal experimental distribution ($0$ OOD warnings).
2. **Load High-Modulus Specimen [OOD]**:
   - $E_{11} = 200.0$ GPa (all other parameters nominal).
   - Generates advisory alert indicating $E_{11}$ exceeds the experimental training range.
3. **Load Deep Notch Specimen [OOD]**:
   - $a_0 = 60.0$ mm (all other parameters nominal).
   - Graph begins strictly at $a[0] = 60.0$ mm and displays an advisory OOD warning.

---

## Automated Consistency Verification

The platform performs real-time automated verification upon each simulation:
- $a[0] = a_0$ strictly verified at initial load.
- Result cards match final graph endpoints with zero discrepancy:
  - $P_{\text{final}} = P(\delta_{\text{max}})$
  - $\delta_{\text{final}} = \delta_{\text{max}}$
  - $C_{\text{final}} = C(\delta_{\text{max}})$
  - $a_{\text{final}} = a(\delta_{\text{max}})$
  - $G_{I, \text{final}} = G_I(\delta_{\text{max}})$
- $C = \delta / P$ strictly satisfied across all 70 discrete points.

---

## License & Disclaimer

AI model predictions are intended for research, surrogate modeling, and virtual testing analysis. Critical aerospace and structural composite designs must be verified with physical physical DCB testing according to ASTM D5528 protocols.
