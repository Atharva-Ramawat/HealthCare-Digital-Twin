# Temporal Baseline Model Definitions & Evaluation Protocols
**Module**: `ml/temporal/baselines.py`  
**Dataset**: MIMIC-IV Demo v2.2 (Pulmonary Cohort)  
**Evaluated Split**: Held-Out Test Set (10 Patients / 11 ICU Stays / 3,147 Sequences)

---

## 1. Moving-Average Trend Forecaster

### Mathematical Formulation
For each sequence ending at index $i$ (time $t$), the historical context spans $[i - 23, i]$ ($t - 5.75\text{h} \dots t$).
For each physiological vital channel $c \in \{\text{HR}, \text{SpO}_2, \text{SBP}, \text{RR}, \text{Temp}\}$, the forecast $\hat{y}_{t+k, c}$ for future steps $k \in \{1, 2, 3, 4\}$ is defined as:
$$\hat{y}_{t+k, c} = \frac{1}{|S_{\text{hist}}|} \sum_{j \in S_{\text{hist}}} x_{j, c}$$
where $S_{\text{hist}}$ contains the last 4 historical observed steps $[i - 3, i]$. If historical observations are unmeasured, the population training median is used.

### Leakage Audit & Protocol Compliance
1. **Historical Data Only**: Uses strictly observations $\le t$. Zero access to future time steps $t+1 \dots t+4$.
2. **Identical Sequence Windows**: Evaluated across all 3,147 held-out test windows.
3. **Identical Masking**: Evaluated strictly on observed cells where `forecast_valid_mask == 1.0` (60,926 valid future points).

---

## 2. Naive Persistence Forecaster

### Formulation
Sets predicted future values equal to the last known historical observation at prediction time $t$:
$$\hat{y}_{t+k, c} = x_{t, c} \quad \forall k \in \{1, 2, 3, 4\}$$

---

## 3. Supervised Machine Learning Baselines

### A. Random Forest Regressor (Forecasting)
- **Input**: Flattened historical sequence tensor $\mathbb{R}^{24 \times 45 = 1080}$.
- **Output**: 20 continuous future vital predictions ($4 \text{ steps} \times 5 \text{ channels}$).
- **Configuration**: 30 estimators, max depth 8, trained on training partition.

### B. Random Forest & Logistic Regression Classifiers (Instability Event)
- **Input**: Flattened historical features ($1,080$ dimensions).
- **Target**: Future Physiological Instability Event ($Y \in \{0, 1\}$).
- **Configuration**: Evaluated using probability thresholds, reporting AUROC, AUPRC, Accuracy, Precision, Recall, and F1.

---

## 4. Test Set Benchmark Summary

| Model | Task | Metric 1 | Metric 2 | Metric 3 |
| :--- | :--- | :---: | :---: | :---: |
| **Moving-Average Trend** | Vital Forecasting | Overall MAE: **1.3707** | Overall RMSE: **3.1485** | $R^2 \ge 0.85$ on all 5 vitals |
| **Naive Persistence** | Vital Forecasting | Overall MAE: **1.7886** | Overall RMSE: **5.9941** | SpO2 $R^2 = 0.3241$ |
| **Random Forest Regressor** | Vital Forecasting | Overall MAE: **4.5482** | Overall RMSE: **8.0337** | Tabular over-smoothing |
| **Random Forest Classifier**| Instability Event | AUROC: **0.8516** | AUPRC: **0.7383** | F1: **0.6075** |
| **Logistic Regression** | Instability Event | AUROC: **0.7347** | AUPRC: **0.5647** | F1: **0.5063** |
