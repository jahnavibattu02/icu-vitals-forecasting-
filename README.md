# ICU Patient Vitals Forecasting (LSTM)

This project builds a time-series forecasting pipeline to predict next-step ICU vital signs (HR, MAP, SpO₂) from past observations using an LSTM model. It includes patient-level splitting to avoid leakage, a naive baseline, and a deployable Streamlit app suitable for Hugging Face Spaces.

## ⚠️ Disclaimer
This project is for educational and research demonstration only. Do not use for clinical decision-making. Do not upload real patient data to public repositories or demos.

## Data Format
CSV columns required:
- `patient_id`
- `timestamp` (numeric or sortable)
- `HR`, `MAP`, `SpO2`

## Quickstart (Local)
```bash
pip install -r requirements.txt
python -m src.make_synthetic_data --out synthetic_icu_vitals.csv
python -m src.train --data_csv synthetic_icu_vitals.csv
streamlit run app.py
```
