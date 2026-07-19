# Submission — Group 19

## Deliverables

| Deliverable | Where |
|---|---|
| Report (full write-up, with figures) | [`report/report.pdf`](report/report.pdf) — also `report/report.md` |
| System simulation video | <https://youtu.be/Wt0Sk_pNswU> |
| GitHub repository | <https://github.com/Hassan-Adelani-Luqman/g19-biometric-product-recommender> |
| Team member contributions | [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) |

## Repository contents (assignment checklist)

| Required item | Location |
|---|---|
| Datasets (raw) | `data/raw/customer_social_profiles.csv`, `data/raw/customer_transactions.csv` |
| Merged dataset (feature-engineered) | `data/processed/merged_dataset.csv` |
| Feature tables | `data/processed/image_features.csv`, `data/processed/audio_features.csv` |
| Pipeline scripts (3 models + CLI app) | `scripts/` (`merge_and_clean`, `image_features`, `audio_features`, `train_face_model`, `train_voice_model`, `train_product_model`, `collect_metrics`, `app.py`) |
| Jupyter notebooks | `notebooks/01_eda_and_merge`, `02_image_pipeline`, `03_audio_pipeline`, `04_product_model` |
| Trained models | `models/face_model.joblib`, `voice_model.joblib`, `product_model.joblib` |
| Metrics | `results/metrics.json` |
| System simulation log | `results/simulation_log.txt` |

## How to run

Full setup and run instructions are in [`README.md`](README.md). The multimodal app:

```
python scripts/app.py --face image_data/Hassan_Adelani_Luqman/neutral.jpg --voice audio_data/Hassan_Adelani_Luqman/confirm_transaction.wav
```
