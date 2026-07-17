# Group 19 — Multimodal Auth & Product Recommendation System

A command-line system that authenticates a customer with **two biometric factors** and then
recommends a product. A face image is checked against a **Facial Recognition Model**; if it
passes, a **Product Recommendation Model** predicts the customer's product from their tabular
profile; a voice clip is then checked against a **Voiceprint Model** that must match the same
identity before the prediction is revealed. Any failed factor prints `Access Denied`.

```
Start → Facial Recognition ──fail──► Access Denied
             │pass
             ▼
        Product Recommendation ──► Voice Validation ──fail──► Access Denied
                                          │pass
                                          ▼
                                  Display Predicted Product
```

## Repository layout

```
g19-biometric-product-recommender/
├── PLAN.md                     # full execution plan — read this first
├── README.md
├── CONTRIBUTIONS.md            # (added in Phase 8)
├── requirements.txt
├── data/
│   ├── raw/                    # exported CSVs (source of truth)
│   └── processed/              # merged_dataset.csv, image_features.csv, audio_features.csv
├── image_data/<member>/        # 3 face photos per member
│   └── impostor/               # non-member photos for the rejection demo
├── audio_data/<member>/        # 2 voice clips per member
├── notebooks/                  # 01_eda_and_merge, 02_image, 03_audio, 04_product
├── scripts/                    # feature extraction, training, collect_metrics, app.py
├── models/                     # trained .joblib artifacts
├── results/
│   ├── metrics/                # per-model metric shards
│   ├── metrics.json            # collected (written only by collect_metrics.py)
│   └── simulation_log.txt
└── report/report.md
```

## Setup

Requires **Python 3.12** (the ML stack — `mediapipe`, `librosa`/`numba` — does not yet ship
wheels for 3.14).

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment notes (Phase 0 findings)

- **Built with Python 3.12.10, not 3.14.** `numba` / `llvmlite` (pulled in by `librosa`) and
  `mediapipe` do not ship 3.14 wheels yet, so a 3.14 venv would break the audio and image
  pipelines. Use `py -3.12`.
- **Face features use MediaPipe + HOG, not `face_recognition`.** We probed `face_recognition`
  in Phase 0; its `dlib` dependency failed to build a wheel on this Windows machine (no working
  CMake/MSVC toolchain). Per PLAN.md §8 Phase 3 this selects **rung 2**: MediaPipe face
  detection + crop, then HOG descriptors via `scikit-image`. Both are installed and import
  cleanly. Vision Lead: build on rung 2 — don't spend time fighting dlib.

Full setup and run instructions are added in Phase 8. Until then, see **PLAN.md** for how the
project is built phase by phase and who owns what.
