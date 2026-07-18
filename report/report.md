# Group 19 — Multimodal Authentication & Product Recommendation System
## Project Report

> All model numbers in this report are read from `results/metrics.json` (produced by
> `scripts/collect_metrics.py`) and the simulation results from `results/simulation_log.txt`.
> They are not restated from memory.

---

## 1. Overview & architecture

The system authenticates a customer with **two biometric factors** and then recommends a
product, following the assignment flowchart exactly:

```
Start → Facial Recognition ──fail──► ACCESS DENIED
             │ pass
             ▼
        Product Recommendation (held) ──► Voice Validation ──fail──► ACCESS DENIED
                                                │ pass
                                                ▼
                                     Display Predicted Product
```

Three independently trained models sit behind this flow:

| Stage | Model | Input |
|---|---|---|
| Facial Recognition | RandomForest on HOG face features | a face image |
| Product Recommendation | LogisticRegression on social features | the member's tabular record |
| Voice Validation | LogisticRegression on MFCC features | a voice `.wav` |

The **product prediction is computed but held** after the face passes, and only revealed if
the voice **both** clears its confidence threshold **and** is recognised as the *same member*
the face was. That identity cross-check (Section 8) is the core of the multimodal design — an
approved voice alone cannot unlock another member's transaction.

---

## 2. Data merge & cleaning (Phase 2)

Two sources were merged: `customer_social_profiles` and `customer_transactions`.

- **Join key.** The tables use different ID schemes — `customer_id_new` (`"A178"`) vs
  `customer_id_legacy` (`178`). Every social ID matches `^A\d+$`, so stripping the leading `A`
  reconciles them. 61 customers appear in both tables.
- **Grain.** Both tables are one-to-many per customer (several social platforms; several
  purchases), so a naive row-join would multiply rows. Social profiles were **aggregated to one
  row per customer** (mean engagement/interest, distinct-platform count, dominant platform and
  sentiment) and then joined onto transactions.
- **Join type — INNER.** The model predicts *from* the social profile, so the population is the
  customers who have a social profile **and** purchased. An inner join avoids fabricating
  predictor features by imputation (which a left join would force on social-less customers).
- **Cleaning.** 5 exact-duplicate social rows dropped; 10 missing `customer_rating` values
  median-imputed; `purchase_date` parsed to datetime; categoricals typed.
- **Leakage demarcation.** The transaction columns (`purchase_amount`, `customer_rating`,
  `purchase_date`) are *post-purchase* — unavailable "when a customer visits social media" — so
  they are excluded from the model. The predictors are the five social features.

Output: `data/processed/merged_dataset.csv` — **117 rows, one per transaction**, target
`product_category`. EDA (see `notebooks/01_eda_and_merge.ipynb`) found `primary_platform` to be
the most informative single feature and numeric correlations to be weak.

---

## 3. Image pipeline & Facial Recognition (Phase 3)

- **Data.** 4 members × 3 expressions (neutral, smiling, surprised) = 12 source images, plus
  impostor photos for threshold validation.
- **Feature extraction — rung 2.** `face_recognition`/`dlib` failed to build on the Windows
  toolchain (Phase 0), so the pipeline uses **MediaPipe** face detection + crop, then
  **scikit-image HOG** descriptors (8100-D). Documented in the metrics as
  *"MediaPipe face crop + scikit-image HOG"*.
- **Augmentation.** 3 per image (rotation, horizontal flip, brightness) → 48 feature rows
  (12 originals + 36 augmented).
- **Model.** RandomForest predicting the member, with a max-probability **confidence threshold**
  (0.4025) selected against the impostor set to derive a known/unknown decision.

---

## 4. Audio pipeline & Voiceprint (Phase 4)

- **Data.** 4 members × 2 phrases ("Yes, approve" / "Confirm transaction") = 8 source clips,
  loaded at 16 kHz mono, silence-trimmed and loudness-normalised.
- **Augmentation.** 3 per clip (pitch shift, time stretch, added noise) → 32 feature rows.
- **Features.** MFCC mean/std, MFCC deltas, spectral roll-off, RMS energy, ZCR and spectral
  centroid are all extracted to `audio_features.csv` (86 columns). The **model uses a lean
  subset** — MFCC mean/std + roll-off + RMS (30 features) — because the full 86-D vector
  over-fits 8 clips. The classifier is a regularised **LogisticRegression**.

---

## 5. Product Recommendation model (Phase 5)

- Trains on the **five social predictors only**, predicting `product_category` (5 classes).
- **LogisticRegression (C=0.2)** on one-hot categoricals + scaled numerics — chosen over
  RandomForest/XGBoost, which over-fit this small, mostly-categorical data.
- Each team member is mapped to a demo customer with a distinct predicted product
  (Gentil→Electronics, Hassan→Sports, Mahlet→Clothing, Yvette→Groceries) so Phase 6 can turn a
  recognised face into a tabular record.

---

## 6. Model results

Read from `results/metrics.json`:

| Model | Split (leakage-safe) | Accuracy | Macro-F1 | Log-loss | Threshold |
|---|---|---|---|---|---|
| **Facial Recognition** | StratifiedGroupKFold(3) by `source_image_id` | **0.979 ± 0.029** | 0.979 | 0.793 | 0.4025 (impostor rejection 100%) |
| **Voiceprint** | phrase-held-out (train `yes_approve` / test `confirm_transaction`) | **0.938** | 0.937 | 0.384 | 0.4516 (accepts 89% of genuine) |
| **Product Recommendation** | StratifiedGroupKFold(5) by `customer_id` | **0.231 ± 0.021** | 0.174 | 1.696 | — (majority baseline 0.222) |

Supporting cross-validation: the voiceprint model's leave-one-clip-out CV is **0.844 ± 0.278**
(high variance is expected at 8 clips). The product model's macro-F1 of 0.174 clearly beats a
majority-class dummy's 0.073, even though its accuracy only edges the 0.222 baseline.

---

## 7. Splitting methodology — why the numbers are honest

Every model is evaluated with a **grouped split**, because augmented copies (image/audio) and a
customer's repeated transactions are near-identical to their source. Splitting by row would put
a source's variants on both sides of the split and inflate the score:

- **Facial Recognition** — grouped by `source_image_id` (all augmentations of one photo stay in
  one fold; the code asserts no group leaks).
- **Voiceprint** — phrase-held-out (train on one phrase, test on the other → text-independent
  speaker verification), plus leave-one-clip-out CV for variance.
- **Product Recommendation** — grouped by `customer_id`. For contrast, a leaky row-split reads
  0.240 vs the honest 0.231; the gap is small here only because the task is intrinsically hard.

This is the report's central methodological claim: the numbers are modest but **trustworthy**.

---

## 8. Multimodal integration & system simulation (Phases 6–7)

`scripts/app.py` loads the three models and runs the flowchart. The **identity cross-check** is
the key multimodal logic: after the face is recognised, the voice must be (a) above its
confidence threshold **and** (b) classified as the *same member*. A high-confidence voice from a
different member is rejected.

The three required scenarios were run through the real CLI and captured to
`results/simulation_log.txt` (via `scripts/run_simulation.py`) — **all three pass**:

| Scenario | Input | Result |
|---|---|---|
| Unauthorised (image) | impostor face | **ACCESS DENIED** at the face step (conf 0.37 < 0.40) |
| Unauthorised (audio) | Hassan's face + Gentil's voice | **ACCESS DENIED** at the voice step (voice = Gentil ≠ face = Hassan) |
| Full valid transaction | Hassan's face + Hassan's voice | **ACCESS GRANTED** → product **Sports** |

---

## 9. Limitations & future work

- **Tiny data.** 12 face images, 8 voice clips, 2 impostor photos, 61 customers. All numbers are
  indicative, not production claims.
- **Facial recognition acceptance.** The 0.979 accuracy is honest out-of-fold, but at the chosen
  threshold only ~67% of *unseen* genuine faces clear the bar (both impostors sit at ~0.37,
  pushing the threshold up). The deployed model — trained on all images — passes the actual
  member photos at 0.88–0.93 confidence, so the demo is robust; a larger, more varied face set
  is the fix.
- **Product recommendation is weak by nature.** The predictors are customer-level while 30 of 61
  customers buy more than one category, so one profile maps to several products — an irreducible
  ceiling. Per-visit/session context (not more model tuning) is what would raise this.
- **Thresholds** are tuned on a handful of impostors / out-of-fold confidences; more negative
  examples would make them more robust.

More data per member — more photos, more voice recordings, richer per-visit signals — is the
single highest-value improvement across all three models.

---

## 10. Contributions

Individual contributions are recorded in `CONTRIBUTIONS.md`. The work was organised by role —
Data, Vision, Audio and Integration leads — and built phase-by-phase on separate branches merged
into `main` via pull requests, so the git history reflects each member's contribution.
