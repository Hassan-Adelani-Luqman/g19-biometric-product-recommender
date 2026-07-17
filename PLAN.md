# Group 19 — Multimodal Auth & Product Recommendation System

**Execution plan.** This document is the single source of truth for how this project gets built. It is written to be handed directly to Claude Code, and to be read by all four team members.

Repo: `g19-biometric-product-recommender`

---

## 1. How to use this document

1. This file lives at the repo root as `PLAN.md`.
2. Start a Claude Code session in the repo root, on the correct branch for the phase you own (see §3).
3. Give it this instruction to begin a phase:

   > Read PLAN.md. Execute Phase N. Stop and summarize what you did, then wait for my confirmation before doing anything else. Do not skip ahead. If a precondition isn't met, stop and tell me — do not guess or fabricate data.

4. Work phases **in order**. Confirm the phase's "Done when" checklist before moving on.
5. Where a phase needs a real file from a human (photos, audio, exported CSVs), Claude Code **pauses and asks** rather than generating placeholder data.

---

## 2. Team & ownership

Roles and responsibilities are fixed by the task distribution below. **Replace `Member N` with real names before Phase 0 starts.** The branch for each phase is named in its Phase heading in §8.

| Role | Member | Responsibilities | Phases |
|---|---|---|---|
| Data Lead | Member 1 | Tabular merge, cleaning, EDA, Product Recommendation Model | 2, 5 |
| Vision Lead | Member 2 | Image collection / augmentation / features, Facial Recognition Model | 3 |
| Audio Lead | Member 3 | Audio collection / augmentation / features, Voiceprint Model | 4 |
| Integration Lead | Member 4 | CLI app, multimodal decision logic, system simulation, GitHub repo structure | 0, 1, 6, 7, 9 |
| All four | — | Report — each member writes up the phase they owned | 8 |

Reading the distribution onto the phases:

- **Data Lead** — Phase 2 is the merge/cleaning/EDA, Phase 5 is the Product Recommendation Model. Sequential, both yours.
- **Vision Lead** — Phase 3 is collection through to the trained Facial Recognition Model, end to end.
- **Audio Lead** — Phase 4 is collection through to the trained Voiceprint Model, end to end.
- **Integration Lead** — "GitHub repo structure" is Phase 0, and Phase 1 (data validation) goes here too since it gates everyone else. The CLI app is Phase 6, multimodal decision logic is the identity cross-check in Phase 6, system simulation is Phase 7. Phase 9 is final QA.

Two things this distribution implies that are easy to miss:

- **Vision and Audio Leads own data collection, not just modelling.** Chasing the other three members for their 3 photos and 2 clips — and the impostor photos in §5 — is inside your role. Start now; it's the long pole and it blocks your own Phase 3/4 work.
- **Phases 2, 3, and 4 run in parallel.** Data, Vision, and Audio Leads are gated on Phase 1, not on each other. The Integration Lead is idle between Phase 1 and Phase 6 — that gap is the natural time to draft the README, write `collect_metrics.py`, and set up branch protection on `main`.

---

## 3. Branching & contribution workflow

### 3.1 Why branches here

Marks are partly individual. `git log` on `main` is the evidence of who did what, so the workflow is built to keep that history legible and attributable.

### 3.2 Branch model

One branch per phase, one owner per branch. Branch off `main`, PR back into `main`.

```
main
 ├─ phase-0-scaffold ─────────────► merge ─┐   (everyone pulls before starting)
 ├─ phase-1-data-validation ──────► merge ─┤
 │                                          │
 │   ┌── phase-2-tabular-eda ──────► merge ─┤   Data Lead
 │   ├── phase-3-image-pipeline ───► merge ─┤   Vision Lead    } run in parallel
 │   └── phase-4-audio-pipeline ───► merge ─┤   Audio Lead
 │                                          │
 ├─ phase-5-product-model ─────────► merge ─┤   (needs Phase 2 on main)
 ├─ phase-6-integration-cli ───────► merge ─┤   (needs 3, 4, 5 on main)
 ├─ phase-7-simulation ────────────► merge ─┤
 ├─ phase-8-report ────────────────► merge ─┤
 └─ phase-9-final-qa ──────────────► merge ─┘
```

The only true parallel window is **Phases 2, 3, 4**. Everything else is gated.

### 3.3 Gates — `main` must be green before these start

| Gate | Requires on `main` |
|---|---|
| Anyone starts Phase 1+ | Phase 0 merged |
| Phases 2, 3, 4 start | Phase 1 merged (data validated) |
| Phase 5 starts | Phase 2 merged |
| Phase 6 starts | Phases 3, 4, **and** 5 all merged |
| Phase 8 starts | Phase 7 merged |

### 3.4 The loop, per phase

```bash
git checkout main
git pull origin main
git checkout -b phase-3-image-pipeline

# ... work, committing as you go ...
git add scripts/image_features.py
git commit -m "Phase 3: HOG feature extraction over augmented image set"

# before opening the PR, bring main in:
git fetch origin
git merge origin/main          # merge, NOT rebase — see 3.6
# resolve any conflicts, re-run your scripts, confirm still green

git push -u origin phase-3-image-pipeline
gh pr create --base main --title "Phase 3: image pipeline" --body "..."
```

Then: **one other member reviews and merges.** Nobody merges their own PR. The reviewer's job is to pull the branch and confirm the scripts actually run, not just to skim the diff.

### 3.5 Merge, don't squash

Use a regular merge commit (`gh pr merge --merge`). **Do not squash.** Squashing collapses each member's individual commits into one, which destroys the per-member contribution history that partly determines individual marks.

### 3.6 Merge, don't rebase

When syncing your branch with `main`, use `git merge origin/main`, not `git pull --rebase`. Rebasing rewrites commits, and rebase conflicts in `.ipynb` files (which are JSON) are genuinely awful to resolve — `--ours` and `--theirs` are inverted during a rebase and it's easy to silently destroy work. Merge is slower-looking in the log and much harder to get wrong.

### 3.7 Conflict-avoidance rules (these matter more than they look)

These exist because Phases 3, 4, and 5 write to overlapping places. Follow them and the parallel window is conflict-free.

| File | Rule |
|---|---|
| `results/metrics.json` | **Never write to it during Phases 3–5.** Each model writes its own shard: `results/metrics/facial_recognition.json`, `results/metrics/voiceprint.json`, `results/metrics/product_recommendation.json`. Only `scripts/collect_metrics.py` (run in Phase 6) merges the shards into `results/metrics.json`. Three people appending to one JSON file is a guaranteed three-way conflict. |
| `requirements.txt` | Pinned **completely in Phase 0**, including image and audio deps. Nobody edits it later. If you genuinely need a new package, say so in the group chat and let the Integration Lead add it on a small dedicated branch. |
| `notebooks/*.ipynb` | One owner each. **Never edit someone else's notebook.** Run `jupyter nbconvert --clear-output --inplace <nb>` before committing to cut diff noise. |
| `models/*.joblib` | Binary — never resolve a conflict by merging. Take either side, then **re-run the training script** to regenerate. |
| `data/processed/*.csv` | Same as models: regenerate, don't merge. |
| Augmented media | Gitignored. Derived from the raw files, regenerable, and would bloat the repo. |

### 3.8 Commit hygiene

- Commit messages: `Phase 2: tabular merge + EDA complete`. Small, real commits inside the phase are encouraged — the history is the contribution evidence.
- Push your branch at least daily so others can see progress rather than reviewing one giant diff at the end.
- No untracked scratch work in the final repo. All code lives under `/scripts` or `/notebooks`.

---

## 4. Critical pitfalls

Four things that will quietly wreck the results if missed. Each is handled in its phase; they're listed here because they're cross-cutting and a grader will probe all four.

1. **Train/test leakage via augmentation** (Phases 3, 4) — augmented copies must never straddle the split. See §8.
2. **The "unknown" class needs real negatives** (Phases 3, 4) — you cannot build an impostor class out of your own members' augmented photos. See §Prerequisites and §8.
3. **dlib/`face_recognition` on Windows** (Phase 3) — often fails to build. Decision ladder in §8.
4. **Tiny n** (Phases 3, 4) — 12 source images, 8 source audio clips. Single-split accuracy on that is noise. Use cross-validation and report mean ± std.

---

## 5. Prerequisites (human tasks, before Phase 0)

Claude Code cannot access private Google Sheets, photograph people, or record audio. Before Phase 0:

- [ ] Export `customer_social_profiles` and `customer_transactions` from Google Sheets as CSV → `data/raw/customer_social_profiles.csv`, `data/raw/customer_transactions.csv`.
- [ ] Each of the 4 members supplies 3 face photos (neutral, smiling, surprised) → `image_data/<member_name>/<member_name>_<expression>.jpg`.
- [ ] Each of the 4 members supplies 2 audio clips ("Yes, approve", "Confirm transaction") → `audio_data/<member_name>/<member_name>_<phrase>.wav`.
- [ ] **Impostor face data (2–3 photos of one non-member).** Phase 7 requires demoing a rejected face. You cannot do this with member photos. Ask a classmate outside the group and get their OK to use it for coursework → `image_data/impostor/`. If nobody's available, tell Claude Code — the fallback is a small sample from a public face dataset (LFW), which must be cited in the report. Sort this out now; it blocks Phase 7, and Phase 7 blocks the demo video.
- [ ] Confirm Python version and package manager (assume Python 3.11+, `pip` + `venv` unless told otherwise).
- [ ] Fill in the names in §2.

Note: the **audio** rejection demo needs no extra recordings — Phase 6's design uses member A's face + member B's voice as the mismatch case. See §8, Phase 6.

---

## 6. Global rules for Claude Code

- **Never fabricate data.** If a required input file is missing, stop and ask.
- Every phase produces **runnable, tested code** — run it and show real output before marking the phase done. Not a plan, not a stub.
- Naming: `snake_case` for files, `<member_name>_<label>.<ext>` for media.
- Metrics go to `results/metrics/<modality>.json` shards (see §3.7), never straight to `results/metrics.json`.
- Prefer well-known libraries over custom implementations: `pandas`, `scikit-learn`, `xgboost`, `opencv-python` / `Pillow`, `mediapipe`, `librosa`, `matplotlib` / `seaborn`.
- Environment is **Windows**. Activate the venv with `venv\Scripts\Activate.ps1` (PowerShell), not `source venv/bin/activate`.

---

## 7. Repo structure (created in Phase 0)

```
g19-biometric-product-recommender/
├── PLAN.md
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                      # original exported CSVs
│   └── processed/                # merged_dataset.csv, image_features.csv, audio_features.csv
├── image_data/
│   ├── <member>/                 # 3 raw photos per member
│   └── impostor/                 # non-member photos for the Phase 7 rejection demo
├── audio_data/<member>/          # 2 raw audio clips per member
├── notebooks/
│   ├── 01_eda_and_merge.ipynb
│   ├── 02_image_pipeline.ipynb
│   ├── 03_audio_pipeline.ipynb
│   └── 04_product_model.ipynb
├── scripts/
│   ├── merge_and_clean.py
│   ├── image_features.py
│   ├── audio_features.py
│   ├── train_face_model.py
│   ├── train_voice_model.py
│   ├── train_product_model.py
│   ├── collect_metrics.py        # merges results/metrics/*.json → results/metrics.json
│   └── app.py                    # CLI simulation
├── models/                       # face_model.joblib, voice_model.joblib, product_model.joblib
├── results/
│   ├── metrics/                  # per-modality shards (committed)
│   ├── metrics.json              # collected — written only by collect_metrics.py
│   └── simulation_log.txt
└── report/
    └── report.md                 # later exported to PDF
```

---

## 8. Phases

### Phase 0 — Repo scaffolding & environment
**Owner:** Integration Lead · **Branch:** `phase-0-scaffold` · **Gate:** none

**Goal:** Folder structure, git, venv, pinned dependencies.

1. Create the directory tree in §7 (`.gitkeep` in otherwise-empty dirs).
2. `.gitignore`: `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*_aug*.jpg`, `*_aug*.wav`, and any augmented-media output dir.
3. `requirements.txt` — pin **everything now**, including image and audio deps, so no later phase has to touch this file: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `opencv-python`, `Pillow`, `mediapipe`, `scikit-image`, `librosa`, `soundfile`, `jupyter`, `joblib`.
4. Create the venv, install, and verify: `python -c "import pandas, sklearn, cv2, librosa, skimage; print('ok')"`.
5. **Probe the face-library decision now, not in Phase 3.** Try `pip install face_recognition`. If it fails (dlib needs CMake + VS C++ Build Tools and frequently won't build on Windows), do **not** burn time fighting it — record the failure in the README and use the ladder in Phase 3. Report which rung we're on.
6. Starter `README.md`: title, one-paragraph description, folder structure.

**Done when:** structure exists, imports succeed, face-library rung recorded, first commit made, PR merged to `main`.

---

### Phase 1 — Data ingestion & validation
**Owner:** Integration Lead · **Branch:** `phase-1-data-validation` · **Gate:** Phase 0 on `main`

**Goal:** Confirm every raw input is present and readable before anyone builds on it.

1. Load both CSVs with pandas; print `.shape`, `.dtypes`, `.head()`, null counts.
2. Walk `image_data/` and `audio_data/`; confirm each member folder has 3 images and 2 audio files; confirm `image_data/impostor/` is populated. Print a summary table (member → file count, missing items flagged).
3. Check files are non-empty and actually decodable — a 0-byte `.wav` or a `.jpg` that's really a HEIC will pass an existence check and fail in Phase 3.
4. If anything is missing, **stop and report exactly what's missing.** Do not proceed.

**Done when:** validation report shows all data present, readable, and decodable. This is the gate for three parallel branches — do not merge it until it's genuinely clean.

---

### Phase 2 — Tabular merge, cleaning & EDA
**Owner:** Data Lead · **Branch:** `phase-2-tabular-eda` · **Gate:** Phase 1 on `main`

**Goal:** `data/processed/merged_dataset.csv` + EDA notebook.

1. In `scripts/merge_and_clean.py`, mirrored in `notebooks/01_eda_and_merge.ipynb`:
   - **Inspect columns to find the join key** — likely a customer/user ID. Do not assume a column name.
   - Handle nulls (impute or drop, with the justification written down) and duplicates.
   - Fix dtypes: dates → datetime, categoricals → category.
   - Merge with an explicit join type, and justify inner vs. left in a markdown cell.
   - Post-merge validation: row count before vs. after, null check on newly joined columns, manual spot-check of a few merged rows.
2. EDA: `.describe()`, variable-type breakdown, and **≥3 labeled plots** — a distribution plot, a boxplot for outliers, a correlation heatmap. Each needs a title, axis labels, and a written interpretation. The interpretation is a rubric line item; a plot with no prose next to it scores nothing.
3. Save to `data/processed/merged_dataset.csv`.

**Done when:** CSV exists and loads cleanly, notebook runs top-to-bottom without errors, 3+ interpreted plots present, PR merged.

---

### Phase 3 — Image pipeline
**Owner:** Vision Lead · **Branch:** `phase-3-image-pipeline` · **Gate:** Phase 1 on `main`

**Goal:** `data/processed/image_features.csv` + trained facial recognition model.

**Feature-extraction ladder** — take the highest rung that installs cleanly, document which and why:
1. `face_recognition` embeddings — only if it installed in Phase 0. Do not fight dlib.
2. **`mediapipe` face detection + crop, then HOG descriptors via `scikit-image`.** Pip-installable on Windows, expected default.
3. OpenCV Haar cascade for detection + HOG. Always works.

**1. `scripts/image_features.py` / `notebooks/02_image_pipeline.ipynb`:**
- Load and display all images per member (sanity-check grid plot).
- **≥2 augmentations per image** (rotation, horizontal flip, grayscale, brightness jitter) via `opencv-python`/`Pillow`. Save with a `_aug1` suffix. These are gitignored — the script regenerates them.
- Extract features per image. Document the rung used.
- Save features + labels to `data/processed/image_features.csv` with columns for member, expression, augmented flag, and **`source_image_id`** — you need that last one for the split.

**2. `scripts/train_face_model.py`:**
- Train a **4-class member classifier** (Logistic Regression or Random Forest), then derive the binary known/unknown decision by thresholding max predicted probability. This satisfies the rubric's "known member vs unknown" *and* gives you an identity for Phase 6's cross-modal check. Validate the threshold against `image_data/impostor/`.
- **Split by `source_image_id`, never by row.** You have 12 source images. An augmented copy is nearly identical to its source — if a rotated `luqman_smiling.jpg` lands in train while the original lands in test, the model has already seen the test set. You get ~100% accuracy that means nothing, and "how did you split?" is the first question a grader asks. Use `StratifiedGroupKFold(n_splits=3)` with `groups=source_image_id` and **report mean ± std across folds**, not a single split — with n=12 a single split is noise.
- Evaluate Accuracy, F1, log-loss. Save to `models/face_model.joblib`.
- Write metrics to **`results/metrics/facial_recognition.json`** (not `metrics.json` — see §3.7).

**Done when:** `image_features.csv` covers every image (original + augmented) and carries `source_image_id`, model trains and evaluates with a grouped CV split, metrics shard written, PR merged.

---

### Phase 4 — Audio pipeline
**Owner:** Audio Lead · **Branch:** `phase-4-audio-pipeline` · **Gate:** Phase 1 on `main`

**Goal:** `data/processed/audio_features.csv` + trained voiceprint model.

**1. `scripts/audio_features.py` / `notebooks/03_audio_pipeline.ipynb`:**
- Load each file with `librosa`; plot **waveform and spectrogram** per sample, labeled per member. Both plots are explicit rubric line items, and both need a written interpretation.
- **≥2 augmentations per sample** (pitch shift, time stretch, background noise) via `librosa`. Gitignored, regenerable.
- Extract MFCCs (mean/std across coefficients), spectral roll-off, energy/RMS.
- Save to `data/processed/audio_features.csv` with member, phrase, augmented flag, and **`source_clip_id`**.

**2. `scripts/train_voice_model.py`:**
- Same design as the face model: **4-class speaker classifier**, binary approved/not-approved derived by thresholding.
- **Split by `source_clip_id`.** Only 8 source clips here — even tighter than images. Two defensible options, pick one and justify it: `StratifiedGroupKFold` grouped by clip, or a **phrase-held-out split** (train on "Yes, approve", test on "Confirm transaction"), which is a clean text-independent speaker-verification setup and reads well in the report.
- Evaluate Accuracy, F1, log-loss. Save to `models/voice_model.joblib`.
- Write metrics to **`results/metrics/voiceprint.json`**.

**Done when:** `audio_features.csv` exists with `source_clip_id`, waveform + spectrogram plots present, labeled, and interpreted, model trains with a grouped split, metrics shard written, PR merged.

---

### Phase 5 — Product recommendation model
**Owner:** Data Lead · **Branch:** `phase-5-product-model` · **Gate:** Phase 2 on `main`

**Goal:** Predict the product a customer would purchase from the merged tabular data.

1. In `scripts/train_product_model.py` / `notebooks/04_product_model.ipynb`:
   - Load `merged_dataset.csv`; encode categoricals, scale numerics, **drop leakage-prone columns** (anything that encodes the purchase itself — e.g. a transaction amount or date that only exists because the purchase happened).
   - Train/test split, train a classifier (Random Forest, Logistic Regression, or XGBoost). Justify the choice.
   - Evaluate Accuracy, F1, log-loss.
   - Save to `models/product_model.joblib`; write **`results/metrics/product_recommendation.json`**.
2. The app needs to look up a member's record by identity. Confirm each of the 4 members maps to a row (or a designated demo customer ID) in the merged dataset, and **write that mapping down** — Phase 6 depends on it and it's an easy thing to discover too late.

**Done when:** model trains cleanly, metrics shard written, artifact saved and reloadable, member→record mapping documented, PR merged.

---

### Phase 6 — Multimodal integration & CLI app
**Owner:** Integration Lead · **Branch:** `phase-6-integration-cli` · **Gate:** Phases 3, 4, **and** 5 all on `main`

**Goal:** `scripts/app.py` replicating the assignment flowchart exactly:

```
Start → Facial Recognition Model
          ├─ Fail → Access Denied
          └─ Pass → Run Product Recommendation Model → Voice Validation Model
                                                          ├─ Fail → Access Denied
                                                          └─ Pass → Display Predicted Product
```

1. First, write and run `scripts/collect_metrics.py` — merges `results/metrics/*.json` into `results/metrics.json`. This is the only thing that writes that file, and Phase 8 reads it.
2. Load the three models from `models/`.
3. CLI flow:
   - Accept a face image path → face model → if below threshold, print `Access Denied` and exit.
   - If pass, **hold the identity**, run the product model on that member's record → hold the prediction, **don't display it yet**.
   - Accept an audio path → voice model → if below threshold **or the predicted speaker ≠ the identity from the face step**, print `Access Denied` and exit.
   - If pass, display the predicted product.
4. **The identity cross-check is the point.** A binary "is this an approved voice" makes the multimodal logic trivial — any approved voice unlocks any face. Checking that the voice matches the *claimed identity from the face step* is a real cross-modal decision, it's what "multimodal logic explained" in the rubric is asking for, and it gives Phase 7 its audio-rejection demo (member A's face + member B's voice) with no extra recordings.
5. Runnable non-interactively: `python scripts/app.py --face <path> --voice <path>`, so the demo can be scripted.

**Done when:** `python scripts/app.py --face <path> --voice <path>` runs end-to-end and prints either `Access Denied` or a product prediction, matching the flowchart exactly; `results/metrics.json` collected; PR merged.

---

### Phase 7 — System simulation & testing
**Owner:** Integration Lead · **Branch:** `phase-7-simulation` · **Gate:** Phase 6 on `main`

**Goal:** Demonstrate the three required scenarios. Capture output to `results/simulation_log.txt`.

1. Run and log:
   - **Unauthorized (image):** impostor face → `Access Denied` at the face step.
   - **Unauthorized (audio):** member A's face + member B's voice → `Access Denied` at the voice step.
   - **Full valid transaction:** member A's face + member A's voice → correct product prediction displayed.
2. Note edge cases and fix bugs in `app.py` or the models before moving on.
3. If a threshold is misclassifying, tune it **against the impostor set and document it** — don't tune it until the demo happens to pass.

**Done when:** all three scenarios run correctly, `results/simulation_log.txt` saved, PR merged. **This is the gate for the demo video** — don't record before it passes.

---

### Phase 8 — Documentation & report
**Owner:** All four · **Branch:** `phase-8-report` · **Gate:** Phase 7 on `main`

This is the one shared branch. Section-per-member to keep conflicts manageable — the person who owned the phase writes its section.

1. `report/report.md`:
   - Approach & architecture overview (reference the flowchart).
   - Data merge & cleaning decisions — from Phase 2. *(Data Lead)*
   - Image pipeline: augmentations, feature-extraction rung and why. *(Vision Lead)*
   - Audio pipeline: augmentations, features. *(Audio Lead)*
   - Model choices and results — **read the numbers from `results/metrics.json`**, never restate from memory.
   - **Splitting methodology** — state that splits were grouped by source image/clip and why. This is a defensible-methodology point; say it explicitly rather than leaving a grader to wonder.
   - System simulation results — from `results/simulation_log.txt`. *(Integration Lead)*
   - Limitations: n=12 images and n=8 clips is tiny, thresholds tuned on a handful of impostors, what more data would buy. Say this plainly — naming the limitation scores better than hoping nobody notices.
2. Update `README.md`: setup (`pip install -r requirements.txt`), how to run each notebook/script, how to run `app.py`.
3. `CONTRIBUTIONS.md` — template with each member's name and a blank line. Humans fill it in.
4. Flag to the team: "Ready to record the demo video — suggested script: [X]", summarizing what to show.

**Done when:** `report.md`, updated `README.md`, `CONTRIBUTIONS.md` all exist; report pulls real numbers from logged results; PR merged.

---

### Phase 9 — Final QA against rubric
**Owner:** Integration Lead · **Branch:** `phase-9-final-qa` · **Gate:** Phase 8 on `main`

Go through each rubric item and print a pass/fail table:

- [ ] EDA: summary stats + ≥3 labeled, interpreted plots
- [ ] Data cleaning: nulls/duplicates handled, types fixed, join justified, post-merge checks shown
- [ ] Images: all 4 members × 3 expressions, consistent naming
- [ ] Image augmentation: ≥2 per image, features saved correctly in `image_features.csv`
- [ ] Audio: 2 phrases × 4 members, waveform + spectrogram plotted and interpreted
- [ ] Audio augmentation: ≥2 per sample, MFCCs + roll-off/energy in `audio_features.csv`
- [ ] All 3 models implemented, trained, functional
- [ ] Each model evaluated (accuracy, F1, loss); multimodal logic explained
- [ ] Full transaction + unauthorized demo both working smoothly
- [ ] Report, notebooks, code, and files clean, named, and documented

Additional self-checks beyond the rubric:
- [ ] Fresh clone + `pip install -r requirements.txt` + run `app.py` works from scratch
- [ ] No leakage: every model's split is grouped by source
- [ ] `results/metrics.json` matches the shards in `results/metrics/`
- [ ] `CONTRIBUTIONS.md` filled in by all four members

Fix anything marked fail. Final commit: `Final: submission ready`.

---

## 9. Command cheatsheet (Windows / PowerShell)

```powershell
# environment
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# start a phase
git checkout main; git pull origin main
git checkout -b phase-N-slug

# sync before PR
git fetch origin; git merge origin/main

# open PR
git push -u origin phase-N-slug
gh pr create --base main --title "Phase N: ..." --body "..."

# reviewer merges (never squash)
gh pr merge --merge

# clear notebook outputs before committing
jupyter nbconvert --clear-output --inplace notebooks/02_image_pipeline.ipynb

# run the app
python scripts/app.py --face image_data/<member>/<member>_neutral.jpg --voice audio_data/<member>/<member>_yes_approve.wav
```

---

## 10. Notes for the team

- Push after every phase so all four can review incrementally rather than reviewing one giant diff at the end.
- Phases 2–5 are the parallel window. If you're the Vision or Audio Lead, don't wait on the Data Lead — you're gated on Phase 1, not on Phase 2.
- Record the demo video only after Phase 7 passes — show the two failure cases and the one success case from the simulation log.
- Sort out the impostor photos in week one. It's the one prerequisite with no software workaround, and it blocks Phase 7 → the video → submission.
