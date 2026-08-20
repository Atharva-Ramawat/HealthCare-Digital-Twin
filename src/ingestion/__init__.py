"""
ingestion package
-----------------
Responsible for loading raw clinical data from supported sources:
  - MIMIC-IV (CSV / BigQuery)
  - MIMIC-III Waveform Database (wfdb)
  - VitalDB (API / CSV)
  - eICU Collaborative Research Database
  - Synthetic generator (fallback / dev mode)
"""
