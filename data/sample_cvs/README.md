# 📄 Sample CVs for Testing & Demonstration

This folder contains curated candidate CVs in both **PDF** and **TXT** formats.

You can drag & drop any `.pdf` or `.txt` file directly into the Streamlit Web UI (`app/streamlit_app.py`) or upload them via the FastAPI endpoint (`POST /api/analyze/file`).

## Matching Guide & Test Scenarios

| File Name | Format | Language | Target Job Description | Expected Match Tier | Expected Score |
| :--- | :---: | :---: | :--- | :--- | :---: |
| `01_Alex_Morgan_Senior_Backend_EN` | PDF & TXT | EN | `../sample_job_descriptions/01_Senior_Backend_Engineer_EN.txt` | Tier 1 | Expected Score: 85-98% |
| `02_Jordan_Taylor_Mid_Backend_EN` | PDF & TXT | EN | `../sample_job_descriptions/01_Senior_Backend_Engineer_EN.txt` | Tier 2 | Expected Score: 60-75% |
| `03_Casey_Bennett_Data_Analyst_Pivot_EN` | PDF & TXT | EN | `../sample_job_descriptions/01_Senior_Backend_Engineer_EN.txt` | Tier 3 | Expected Score: 35-50% |
| `04_Riley_Chen_Graphic_Designer_Mismatch_EN` | PDF & TXT | EN | `../sample_job_descriptions/01_Senior_Backend_Engineer_EN.txt` | Tier 4 | Expected Score: < 25% |
| `05_Maxime_Girard_Ingenieur_MLOps_FR` | PDF & TXT | FR | `../sample_job_descriptions/02_Ingenieur_MLOps_FR.txt` | Tier 1 | Expected Score: 85-98% |
| `06_Camille_Rousseau_Junior_Data_Scientist_FR` | PDF & TXT | FR | `../sample_job_descriptions/02_Ingenieur_MLOps_FR.txt` | Tier 2 | Expected Score: 60-75% |
| `07_Lucas_Moreau_Admin_Systeme_Pivot_FR` | PDF & TXT | FR | `../sample_job_descriptions/02_Ingenieur_MLOps_FR.txt` | Tier 3 | Expected Score: 35-50% |
| `08_Sophie_Lefevre_Juriste_Mismatch_FR` | PDF & TXT | FR | `../sample_job_descriptions/02_Ingenieur_MLOps_FR.txt` | Tier 4 | Expected Score: < 25% |
| `09_Alex_Morgan_Multi_Column_ATS_Warning_EN.pdf` | PDF | EN | `../sample_job_descriptions/01_Senior_Backend_Engineer_EN.txt` | Layout Test | Triggers ATS Multi-Column Warning |
