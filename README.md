# Group 19 — Multimodal Auth & Product Recommendation System

A command-line system that authenticates a customer with **two biometric factors** and then
recommends a product. A face image is checked against a **Facial Recognition Model**; if it
passes, a **Product Recommendation Model** predicts the customer's product from their tabular
profile; a voice clip is then checked against a **Voiceprint Model** that must match the same
identity before the prediction is revealed. Any failed factor prints `ACCESS DENIED`.

```
Start → Facial Recognition ──fail──► ACCESS DENIED
             │ pass
             ▼
        Product Recommendation ──► Voice Validation ──fail──► ACCESS DENIED
                                          │ pass
                                          ▼
                                  Display Predicted Product
```

## Repository layout

```
g19-biometric-product-recommender/
├── PLAN.md                     # full phase-by-phase execution plan
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
├── data/
│   ├── raw/                    # exported CSVs
│   └── processed/              # merged_dataset.csv, image_features.csv, audio_features.csv
├── image_data/<member>/        # 3 face photos per member (+ impostor/)
├── audio_data/<member>/        # 2 voice clips per member (.wav)
├── notebooks/                  # 01 eda+merge, 02 image, 03 audio, 04 product
├── scripts/                    # feature extraction, training, app.py, simulation
├── models/                     # face_model / voice_model / product_model .joblib
├── results/
│   ├── metrics/                # per-model metric shards
│   ├── metrics.json            # collected (single source of truth for the report)
│   └── simulation_log.txt
└── report/report.md
```

## Setup

Requires **Python 3.12**.

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment notes (Phase 0 findings)

- **Use Python 3.12, not 3.14.** `numba`/`llvmlite` (via `librosa`) and `mediapipe` do not ship
  3.14 wheels yet, so a 3.14 venv breaks the audio and image pipelines.
- **Face features use MediaPipe + HOG, not `face_recognition`.** `dlib` failed to build a wheel
  on the Windows toolchain, so the pipeline uses MediaPipe face detection + crop then
  scikit-image HOG descriptors. Both install and import cleanly.

## How to run

All commands are run from the repo root with the venv active.

**1. Validate the raw data**
```powershell
python scripts/validate_data.py
```

**2. Build the processed datasets**
```powershell
python scripts/merge_and_clean.py       # -> data/processed/merged_dataset.csv
python scripts/image_features.py        # -> data/processed/image_features.csv (downloads the MediaPipe detector once)
python scripts/audio_features.py        # -> data/processed/audio_features.csv
```

**3. Train the three models**
```powershell
python scripts/train_face_model.py      # -> models/face_model.joblib      + results/metrics/facial_recognition.json
python scripts/train_voice_model.py     # -> models/voice_model.joblib     + results/metrics/voiceprint.json
python scripts/train_product_model.py   # -> models/product_model.joblib   + results/metrics/product_recommendation.json
```

**4. Collect the metrics into one file**
```powershell
python scripts/collect_metrics.py       # -> results/metrics.json
```

**5. Run the multimodal app**
```powershell
# non-interactive (scriptable for the demo):
python scripts/app.py --face image_data/Hassan_Adelani_Luqman/neutral.jpg --voice audio_data/Hassan_Adelani_Luqman/confirm_transaction.wav

# interactive (prompts for the two paths):
python scripts/app.py
```
Exit codes: `0` = access granted, `1` = access denied, `3` = bad input file.

**6. Reproduce the system simulation**
```powershell
python scripts/run_simulation.py        # runs the 3 scenarios -> results/simulation_log.txt
```

**Notebooks** — launch Jupyter and open any of the four analysis notebooks:
```powershell
jupyter notebook notebooks/
```

## Results & report

- `results/metrics.json` — collected metrics for all three models.
- `results/simulation_log.txt` — the three authentication scenarios.
- `report/report.md` — the full write-up (approach, decisions, results, limitations).

## Team

See `CONTRIBUTIONS.md`. Built by four members across Data, Vision, Audio and Integration roles;
each phase was developed on its own branch and merged into `main` via pull request.
