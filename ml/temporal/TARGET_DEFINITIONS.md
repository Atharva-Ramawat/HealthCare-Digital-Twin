# MIMIC-IV TEMPORAL TARGET SPECIFICATIONS & CLINICAL DEFINITIONS
**Module**: `ml/temporal/targets.py`  
**Dataset**: MIMIC-IV Clinical Database Demo v2.2  
**Domain**: Pulmonary & Critical Care Medicine  
**Sampling Grid**: 15-minute uniform timeline ($L=24$ historical steps = 6.0 hours)

---

## 1. Head 1 — Physiological Deterioration Target ($Y_{\text{det}} \in \{0, 1\}$)

### Clinical Rationale
In pulmonary ICU care, early warning of impending respiratory failure, hemodynamic collapse, or shock is critical. Rather than predicting distal hospital mortality, this target predicts **short-term future physiological decompensation**.

### Specification
- **Input Context Window**: $[t - 23, t]$ (24 consecutive 15-minute observations = 6.0 hours).
- **Prediction Point**: $t$.
- **Future Forecast Window**: $[t + 1, t + 4]$ (next 1.0 hour = 4 steps of 15 minutes).
- **Exact Formulation Rule**:
  $Y_{\text{det}} = 1$ if **sustained abnormality** ($\ge 2$ consecutive or cumulative steps in the future 4-step window) occurs in any of the following physiological channels:
  1. **Severe Hypoxemia**: $\text{SpO}_2 < 90\%$
  2. **Severe Hypotension / Shock State**: $\text{MAP} < 65\text{ mmHg}$ or $\text{SBP} < 90\text{ mmHg}$
  3. **Severe Tachypnea**: $\text{RR} > 28\text{ breaths/min}$
  4. **Extreme Hemodynamic Instability**: $\text{HR} < 45\text{ BPM}$ or $\text{HR} > 130\text{ BPM}$
  Otherwise, $Y_{\text{det}} = 0$.

### Scientific Safeguards & Limitations
- **Persistence Criterion**: Requiring $\ge 2$ abnormal steps filters out transient sensor artifacts (e.g. pulse oximeter movement disconnect).
- **Target Leakage Prevention**: NEWS 2 is strictly excluded from input features so the model must learn intrinsic physiological dynamics.
- **Academic Disclaimer**: This is an academic research prototype estimating physiological trajectory patterns from historical electronic health records.

---

## 2. Head 2 — Risk Tier Target ($Y_{\text{tier}} \in \{0, 1, 2\}$)

### Specification
- **Classes**:
  - `0`: **Low Risk** (Normal physiological bounds throughout future window)
  - `1`: **Medium Risk** (Mild physiological perturbation, e.g. $\text{SpO}_2 \in [90\%, 93\%]$ or $\text{RR} \in [22, 28]$)
  - `2`: **High Risk** (Severe multi-system physiological compromise matching $Y_{\text{det}} = 1$)

---

## 3. Head 3 — Short-Term Physiological Vital Forecasting ($Y_{\text{fore}} \in \mathbb{R}^{4 \times 5}$)

### Specification
- **Target Channels (5)**:
  1. Heart Rate (BPM)
  2. $\text{SpO}_2$ (%)
  3. Systolic Blood Pressure (mmHg)
  4. Respiratory Rate (breaths/min)
  5. Body Temperature (°C)
- **Forecast Steps (4)**:
  - Step 1: $t + 15\text{ min}$
  - Step 2: $t + 30\text{ min}$
  - Step 3: $t + 45\text{ min}$
  - Step 4: $t + 60\text{ min}$
- **Target Tensor Shape**: `(Batch, 4, 5)`
- **Evaluation Metrics**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) benchmarked against Naive Persistence baselines.

---

## 4. Head 4 — Observational Treatment-Response State ($Y_{\text{resp}} \in \{0, 1, 2\}$)

### Specification & Protocol
- **Classes**:
  - `0`: **Stable** (Trajectory change within normal bounds)
  - `1`: **Improving** (Physiological stabilization following treatment initiation)
  - `2`: **Worsening** (Escalating physiological instability following treatment initiation)
- **Evaluation Protocol**:
  - Triggered at treatment initiation events $T_0$ (e.g. Vasopressor infusion, Diuretic, Antibiotic, Bronchodilator, Corticosteroid).
  - Pre-intervention baseline: $[T_0 - 4, T_0]$ (1.0 hour prior to intervention).
  - Post-intervention evaluation window: $[T_0 + 1, T_0 + 8]$ (2.0 hours post-intervention).
  - Composite physiological instability score computed pre vs post:
    $$\Delta = \text{Risk}_{\text{post}} - \text{Risk}_{\text{pre}}$$
    - Improving ($Y_{\text{resp}} = 1$): $\Delta \le -0.15$
    - Worsening ($Y_{\text{resp}} = 2$): $\Delta \ge +0.15$
    - Stable ($Y_{\text{resp}} = 0$): $-0.15 < \Delta < +0.15$

### Strict Scientific Constraints
- **Observational Trajectory Only**: This target quantifies post-intervention physiological trajectories. It does **NOT** estimate counterfactual treatment effect or causal drug efficacy.
