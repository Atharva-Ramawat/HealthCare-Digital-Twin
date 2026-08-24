# MIMIC-IV TEMPORAL TARGET SPECIFICATIONS & CLINICAL DEFINITIONS
**Module**: `ml/temporal/targets.py`  
**Dataset**: MIMIC-IV Clinical Database Demo v2.2  
**Domain**: Pulmonary & Critical Care Medicine  
**Sampling Grid**: 15-minute uniform timeline ($L=24$ historical steps = 6.0 hours)

---

## 1. Head 1 — Future Physiological Instability Event ($Y_{\text{instab}} \in \{0, 1\}$)

### Formal Nomenclature & Semantics
- **Target Name**: **Future Physiological Instability Event** (informally referenced as deterioration proxy).
- **Semantics**: This target serves as a **physiologically-defined empirical proxy** for acute cardiopulmonary instability. It identifies whether sustained vital sign threshold breaches occur within the immediate 1-hour future observation window. It is an EHR trajectory proxy and **not** an automated diagnostic label.

### Specification
- **Input Context Window**: $[t - 23, t]$ (24 consecutive 15-minute observations = 6.0 hours historical context).
- **Prediction Point**: $t$.
- **Future Forecast Window**: $[t + 1, t + 4]$ (next 1.0 hour = 4 steps of 15 minutes).
- **Exact Formulation Rule**:
  $Y_{\text{instab}} = 1$ if **sustained physiological abnormality** ($\ge 2$ consecutive or cumulative steps in the future 4-step window) occurs in any of the following physiological channels:
  1. **Severe Hypoxemia**: $\text{SpO}_2 < 90\%$
  2. **Severe Hypotension / Shock State**: $\text{MAP} < 65\text{ mmHg}$ or $\text{SBP} < 90\text{ mmHg}$
  3. **Severe Tachypnea**: $\text{RR} > 28\text{ breaths/min}$
  4. **Extreme Hemodynamic Instability**: $\text{HR} < 45\text{ BPM}$ or $\text{HR} > 130\text{ BPM}$
  Otherwise, $Y_{\text{instab}} = 0$.

### Target Validity Mask (`deterioration_valid_mask`)
- Evaluated as `1.0` if at least 1 actual bedside vital observation exists in the future window; `0.0` if future intervals are unobserved.

---

## 2. Head 2 — Risk Tier Target ($Y_{\text{tier}} \in \{0, 1, 2\}$)

### Specification
- **Classes**:
  - `0`: **Low Risk** (Normal physiological bounds throughout future window)
  - `1`: **Medium Risk** (Mild physiological perturbation, e.g. $\text{SpO}_2 \in [90\%, 93\%]$ or $\text{RR} \in [22, 28]$)
  - `2`: **High Risk** (Severe multi-system physiological compromise matching $Y_{\text{instab}} = 1$)

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
- **Target Validity Mask (`forecast_valid_mask` $\in \{0, 1\}^{4 \times 5}$)**:
  - Zero future imputation is applied.
  - Loss and evaluation are computed strictly where `forecast_valid_mask == 1.0` (753,981 valid points across all splits).

---

## 4. Head 4 — Event-Linked Observational Treatment-Response State ($Y_{\text{resp}} \in \{0, 1, 2\}$)

### Specification & Protocol
- **Classes**:
  - `0`: **Stable** (Trajectory change within normal bounds)
  - `1`: **Improving** (Physiological stabilization following treatment initiation)
  - `2`: **Worsening** (Escalating physiological instability following treatment initiation)
- **Event Linkage Rule**:
  - Valid **ONLY** at discrete medication initiation timestamps $T_0$ (2,428 unique sequence windows corresponding to 2,438 drug initiation events).
  - Routine monitoring windows without medication initiation receive `response_valid_mask = 0.0`.
- **Composite Instability Shift**:
  $$\Delta = \text{Risk}_{\text{post}} - \text{Risk}_{\text{pre}}$$
  - Improving ($Y_{\text{resp}} = 1$): $\Delta \le -0.15$
  - Worsening ($Y_{\text{resp}} = 2$): $\Delta \ge +0.15$
  - Stable ($Y_{\text{resp}} = 0$): $-0.15 < \Delta < +0.15$

### Strict Scientific Constraints
- **Observational Trajectory Only**: Quantifies observed trajectory shifts following drug initiation. It does **NOT** estimate counterfactual treatment effect or causal drug efficacy.
