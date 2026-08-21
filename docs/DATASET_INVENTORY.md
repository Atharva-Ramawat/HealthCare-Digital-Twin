# Dataset Inventory
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare

**Phase 1 Status:** Complete  
**Last Updated:** August 2026

---

## Available Datasets

### 1. MIMIC-IV Clinical Database Demo v2.2 ✅ LOCALLY AVAILABLE

| Property | Value |
|---|---|
| **Status** | ✅ Downloaded — 28/28 tables |
| **Local path** | `data/raw/mimic_iv_demo/` |
| **Patients** | 100 (empirically confirmed) |
| **ICU stays** | 140 (empirically confirmed) |
| **Hospital admissions** | 275 (empirically confirmed) |
| **Size (key tables)** | chartevents: 5.3 MB, labevents: 1.9 MB |
| **License** | PhysioNet Credentialed Health Data License |
| **Access** | Free PhysioNet account (no CITI required) |
| **Download URL** | https://physionet.org/content/mimic-iv-demo/2.2/ |
| **Role** | **Primary development dataset (Phase 1)** |

**Available tables:**

| Module | Tables |
|---|---|
| `hosp/` | patients, admissions, labevents, d_labitems, diagnoses_icd, d_icd_diagnoses, procedures_icd, prescriptions, pharmacy, microbiologyevents, emar, emar_detail, hcpcsevents, drgcodes, omr, poe, poe_detail, provider, services, transfers |
| `icu/` | chartevents, d_items, datetimeevents, icustays, ingredientevents, inputevents, outputevents, procedureevents |

---

### 2. MIMIC-IV (Full) ❌ NOT AVAILABLE LOCALLY

| Property | Value |
|---|---|
| **Status** | ❌ Not downloaded — requires credentialing |
| **Local path** | `data/raw/mimic_iv/` (empty) |
| **Estimated ICU stays** | ~70,000+ |
| **Access** | PhysioNet account + CITI training |
| **Download URL** | https://physionet.org/content/mimiciv/ |
| **Role** | **Primary research dataset (Phase 2+)** |

To obtain full MIMIC-IV:
1. Register at https://physionet.org/
2. Complete CITI "Data or Specimens Only Research" training
3. Sign the PhysioNet Credentialed Health Data Use Agreement
4. Download to `data/raw/mimic_iv/`

---

### 3. VitalDB ❌ NOT DOWNLOADED (Phase 1 Stub Only)

| Property | Value |
|---|---|
| **Status** | ❌ Not downloaded (Phase 1 decision: stub only) |
| **Local path** | `data/raw/vitaldb/` (empty) |
| **Content** | Surgical ICU waveform data (~1s resolution) |
| **Access** | `pip install vitaldb` (API access) |
| **Role** | Optional high-frequency replay validation |

**Phase 1 decision:** VitalDB is NOT a Phase 1 dependency. If needed for sub-minute replay validation, access via the `vitaldb` Python package in a later phase.

---

### 4. eICU Collaborative Research Database ❌ NOT AVAILABLE

| Property | Value |
|---|---|
| **Status** | ❌ Not downloaded — requires credentialing |
| **Local path** | `data/raw/eicu/` (empty) |
| **Content** | Multi-centre US ICU data (~200 hospitals) |
| **Access** | PhysioNet credentialing |
| **Role** | Optional external validation dataset |

---

## Dataset Priority Order

1. **MIMIC-IV Demo** — Phase 1 development (available now)
2. **MIMIC-IV full** — Phase 2+ research training (requires credentialing)
3. **eICU** — Optional external validation (requires credentialing)
4. **VitalDB** — Optional high-frequency source (no credentialing needed)
