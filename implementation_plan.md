# B.Tech Final Year Project — Implementation Plan
## Neuro-Fuzzy Inferencing Based Low-Light Face Super-Resolution System

> **Target**: 10.0 / 10.0 GPA | **Deadline**: May 10th | **GPU**: Google Colab | **Dataset**: CelebA (standard)
> ⚡ **6-DAY SPRINT — Execution starts immediately. No more planning.**

---

## Project Identity

| Item | Details |
|---|---|
| **Full Title** | Neuro-Fuzzy Inferencing Based System for Low-Light Face Image Super-Resolution with Locality Constrained Representation |
| **Short Title** | ANFIS-LLFSR |
| **Primary Contribution** | Novel pipeline combining ANFIS-based darkness factor estimation + locality-constrained dictionary learning + regression-based face reconstruction under low-light, noisy, and motion-blurred conditions |
| **Referenced Papers** | 5 (detailed below) |
| **Stack** | Python 3.11, PyTorch 2.x, scikit-fuzzy, NumPy/SciPy, FastAPI, React/Vite |
| **Dataset** | CelebA-HQ (primary), FFHQ subset (validation), LFW (face recognition eval) |

---

## Paper-to-Implementation Mapping

This is the academic heart of the project. **Every module below directly implements a technique from one of your 5 papers.**

| Paper | Core Technique | Module We Build |
|---|---|---|
| *Low-light robust face image SR via neuro-fuzzy inferencing-based locality constrained representation* | ANFIS + Locality Constrained Representation (LCR) | `anfis_lcr.py` |
| *Neuro Fuzzy Inferencing Based System and Method For Improving Quality of Dark and Low Resolution Images* | ANFIS system design, membership functions, rule base | `anfis_core.py` |
| *Estimation of darkness factor from low-light images based on adaptive neuro-fuzzy inferencing technique* | Darkness factor (DF) estimation via ANFIS | `darkness_estimator.py` |
| *A new face reconstruction technique for noisy LR images using regression learning* | Ridge/kernel regression on face patches | `regression_reconstructor.py` |
| *Robust face hallucination algorithm using motion blur embedded nearest proximate patch representation* | Motion blur kernel estimation + patch matching | `motion_blur_handler.py` |

> [!IMPORTANT]
> The current codebase uses Zero-DCE + RRDB (GAN-based) — these are **state-of-the-art neural replacements** for your core techniques. They are NOT what the papers describe. The new modules above are what differentiate this as YOUR research project, not an off-the-shelf implementation.

---

## Current Codebase Audit

### ✅ What Exists (Keep & Integrate)

| File | Status | Role in Final System |
|---|---|---|
| `zero_Dce.py` | ✅ Good | Stage 1 pre-enhancement (complement ANFIS) |
| `rrdb_.py` | ✅ Good | Stage 3 super-resolution upsampler |
| `arcface.py` | ✅ Good | Identity preservation loss + face recognition eval |
| `landmark_detector.py` | ✅ Good | Landmark consistency loss |
| `preprocessing.py` | ✅ Good | Degradation pipeline (darkening, blur, noise) |
| `dataset.py` | ✅ Good | DataLoader infrastructure |
| `metrics.py` | ✅ Good | PSNR/SSIM/LPIPS/Face Acc evaluation |
| `visualization.py` | ✅ Good | Training curves |
| `server.py` | ⚠️ Skeleton | Needs image upload + enhance endpoint |
| `app/frontend/` | ⚠️ Empty Vite | Needs full upload UI |
| `app/backend/training/` | ❌ Missing | Training loops needed |

### ❌ What is Missing (What We Will Build)

These are the **academically critical** new components:

```
anfis_core.py            — ANFIS engine (membership functions, rule base, backprop learning)
darkness_estimator.py    — ANFIS-based darkness factor estimation from Paper 3
anfis_lcr.py             — Locality Constrained Representation dictionary (Paper 1)
regression_reconstructor.py  — Patch regression face reconstruction (Paper 4)
motion_blur_handler.py   — Blur kernel estimation + nearest proximate patch (Paper 5)
training/train_anfis.py  — ANFIS training script
training/train_full.py   — End-to-end pipeline training
```

---

## Architecture Overview

```
INPUT (Low-light, Low-res, Possibly Blurred Face)
         │
         ▼
┌─────────────────────────────┐
│  STAGE 1: Darkness Analysis  │  ← Paper 3 (ANFIS darkness factor estimation)
│  darkness_estimator.py       │
│  • Extract illumination stats│
│  • ANFIS → Darkness Factor   │
│  • Adaptive curve correction │
└────────────┬────────────────┘
             │ DF-corrected image
             ▼
┌─────────────────────────────┐
│  STAGE 2: Degradation Fix   │  ← Paper 5 (motion blur) + Zero-DCE (existing)
│  motion_blur_handler.py     │
│  • Blind blur kernel est.   │
│  • Nearest proximate patch  │
│  • Wiener deconvolution     │
└────────────┬────────────────┘
             │ deblurred + enhanced image
             ▼
┌─────────────────────────────┐
│  STAGE 3: Face Hallucination│  ← Papers 1, 4 (ANFIS-LCR + Regression)
│  anfis_lcr.py               │
│  regression_reconstructor.py│
│  • Build face dictionary    │
│  • LCR sparse coding        │
│  • Patch regression         │
│  • ANFIS-guided weighting   │
└────────────┬────────────────┘
             │ reconstructed HR face
             ▼
┌─────────────────────────────┐
│  STAGE 4: Neural Refinement │  ← RRDB (existing, Paper 2 neural component)
│  rrdb_.py                   │
│  • 4× upsampling            │
│  • Detail synthesis         │
│  • Identity preservation    │
└────────────┬────────────────┘
             │
             ▼
        OUTPUT (High-res, well-lit, identity-preserving face)
```

---

## Phased Execution Roadmap

## ⚡ 6-Day Sprint Schedule

| Day | Date | Focus | Deliverable |
|---|---|---|---|
| **Day 1** | May 4 (today) | Dataset + ANFIS Core | `anfis_core.py`, `darkness_estimator.py`, data download script |
| **Day 2** | May 5 | LCR + Regression | `anfis_lcr.py`, `regression_reconstructor.py` |
| **Day 3** | May 6 | Integration + Colab Training | Updated `inference.py`, Colab notebook, training starts |
| **Day 4** | May 7 | Motion Blur + Backend API | `motion_blur_handler.py`, `/api/enhance` endpoint |
| **Day 5** | May 8 | Frontend UI | Upload, Results, Pipeline Visualizer pages |
| **Day 6** | May 9 | Results + Report | Metrics table, visuals, `docs/REPORT.md` draft |
| **Submission** | May 10 | Buffer + Submit | Final polish, zip, submit |

---

## Dataset: CelebA (Standard)

> **Why CelebA instead of CelebA-HQ**: CelebA-HQ is 25GB and requires manual approval. Standard CelebA is ~1.3GB, downloadable in minutes via `gdown`, 202,599 aligned face images at 178×218px — more than sufficient.

```python
# scripts/download_data.py — runs in ~5 minutes
import gdown, zipfile, os

# CelebA aligned images (official Google Drive)
url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
gdown.download(url, 'data/celeba.zip', quiet=False)

with zipfile.ZipFile('data/celeba.zip', 'r') as zf:
    zf.extractall('data/')
# Result: data/img_align_celeba/ with 202,599 images
# Use first 8,000 for train, next 1,000 for val
```

### Phase 0 — Environment & Dataset Setup (Day 1, ~2 hours)

**Goal**: Get a working development environment and data pipeline.

#### Tasks
- [ ] Add `scikit-fuzzy`, `scipy` to `requirements.txt`
- [ ] Write `scripts/download_data.py` (CelebA via gdown)
- [ ] Verify `preprocessing.py` produces correct LR/HR pairs on 10 sample images

#### Deliverables
- Working `venv` with all dependencies
- `data/img_align_celeba/` downloaded (8k train, 1k val split)
- Confirmed LR image looks dark, blurry, and low-res as expected

---

### Phase 1 — ANFIS Core Engine (Days 3–5) ← **Most Important Phase**

**Goal**: Hand-implement the Adaptive Neuro-Fuzzy Inference System from scratch. This is what makes your project academically original.

#### What is ANFIS (for your report)
ANFIS is a 5-layer feed-forward network that maps inputs to outputs via fuzzy IF-THEN rules (Takagi-Sugeno type). It combines:
- **Fuzzy logic**: handles uncertainty in darkness measurement
- **Neural learning**: backpropagation tunes membership function parameters
- **Hybrid learning**: LSE for consequent parameters + gradient descent for premise parameters

#### File: `anfis_core.py` [NEW]

**Layer structure to implement:**
```
Layer 1: Fuzzification — Gaussian/Bell membership functions per input
Layer 2: Rule activation — product of MFs (firing strengths)
Layer 3: Normalization — normalize firing strengths
Layer 4: Consequent — linear functions weighted by normalized strengths
Layer 5: Defuzzification — sum of consequent outputs
```

**Parameters to learn:**
- `{c, σ}` for Gaussian MFs (premise parameters) — learned via gradient descent
- `{p, q, r}` linear consequent parameters — learned via Least Squares Estimation (LSE)

> [!IMPORTANT]
> Do NOT use `scikit-fuzzy` as a black box. Implement the ANFIS layers as a `torch.nn.Module` so that the hybrid learning algorithm is visible, testable, and you can explain every line to your examiner. Use `scikit-fuzzy` only to visualize membership functions.

#### File: `darkness_estimator.py` [NEW]

Based directly on **Paper 3**:
- **Inputs to ANFIS**: Mean pixel intensity (μ), Standard deviation (σ), Dark channel prior (DCP), Entropy (H)
- **Output**: Darkness Factor (DF) ∈ [0, 1] — 0 = well-lit, 1 = extremely dark
- **Training data**: Synthetic — generate images at known gamma values, compute DF ground truth as the gamma parameter itself
- **Use**: DF gates how aggressively Stage 1 correction is applied

**Key formulas to implement from Paper 3:**
```python
# Dark Channel Prior
def dark_channel_prior(image, patch_size=15):
    # min filter over patch, then min over channels
    ...

# ANFIS input feature vector  
def extract_illumination_features(image):
    return [mean_intensity, std_intensity, dark_channel_score, entropy]

# Darkness factor output
darkness_factor = anfis_model(features)  # ∈ [0, 1]
```

---

### Phase 2 — Locality Constrained Representation (Days 6–8)

**Goal**: Implement dictionary-based patch reconstruction from Paper 1.

#### File: `anfis_lcr.py` [NEW]

**LCR (Locality Constrained Representation):**
- Replaces standard sparse coding (expensive L0/L1 minimization)
- Instead: each patch is represented as a **locally weighted linear combination** of nearby dictionary atoms
- "Locality" means atoms close to the patch in feature space get higher weight

**Implementation Steps:**
```python
class LocalityConstrainedDictionary:
    def __init__(self, n_atoms=512, patch_size=8):
        self.D = None  # Dictionary D ∈ R^(d × K)
    
    def build_dictionary(self, hr_patches):
        # 1. Sample HR patches from training set
        # 2. Run K-means to get initial atoms
        # 3. Refine via dictionary learning (K-SVD or MOD)
        ...
    
    def encode_patch(self, lr_patch, beta=1e-4):
        # LCR optimization:
        # min ||patch - D*alpha||^2 + lambda * ||diag(d) * alpha||^2
        # where d_k = exp(||patch - D[:,k]||^2 / beta)
        # Closed-form: alpha = (D^T*D + lambda*diag(d^2))^{-1} * D^T * patch
        ...
    
    def anfis_weight(self, alpha, darkness_factor):
        # ANFIS-guided reweighting of LCR coefficients
        # based on local darkness factor
        ...
```

**ANFIS integration (Paper 1 novelty):**
- The ANFIS from Phase 1 re-weights the LCR coefficients based on local patch darkness
- Dark patches → favor dictionary atoms from similarly dark training examples
- This is the **core contribution** of Paper 1

---

### Phase 3 — Regression Reconstructor (Days 9–10)

**Goal**: Implement Paper 4's regression-based face reconstruction.

#### File: `regression_reconstructor.py` [NEW]

Paper 4 uses **position-patch regression**: for each spatial position in the face, train a local regressor that maps LR patch features → HR patch.

**Implementation:**
```python
class PositionPatchRegressor:
    def __init__(self, n_positions, patch_size=8, method='kernel_ridge'):
        # One regressor per face position
        self.regressors = [KernelRidgeRegressor() for _ in range(n_positions)]
    
    def train(self, lr_patches_by_position, hr_patches_by_position):
        # For each position, fit:
        # HR_patch = W * phi(LR_patch) + b
        # where phi is RBF kernel mapping
        for pos_idx, (lr, hr) in enumerate(zip(lr_patches_by_position, hr_patches_by_position)):
            self.regressors[pos_idx].fit(lr, hr)
    
    def reconstruct(self, lr_face):
        # Extract patches at each position
        # Apply corresponding regressor
        # Blend overlapping patches (Poisson blending or weighted avg)
        ...
```

**Why this matters for your grade**: This is a purely mathematical technique (no deep learning). Your examiner can ask you to derive the ridge regression closed-form solution: `W = (X^T X + λI)^{-1} X^T Y`, and you should be able to explain it on the board.

---

### Phase 4 — Motion Blur Handler (Days 11–12)

**Goal**: Implement Paper 5's motion blur embedded nearest proximate patch representation.

#### File: `motion_blur_handler.py` [NEW]

**Blind Motion Blur Kernel Estimation:**
```python
def estimate_blur_kernel(image, kernel_size=25):
    # Method: Radon transform-based angle detection
    # 1. Edge map via Canny
    # 2. Radon transform → detect dominant blur direction
    # 3. Estimate blur length via power spectrum analysis
    # Returns: (angle, length) → construct motion blur kernel
    ...

def wiener_deconvolution(blurred, kernel, snr=0.01):
    # Frequency domain deconvolution
    # H = FFT(kernel), G = FFT(blurred)
    # F_hat = (H* / (|H|^2 + snr)) * G
    ...
```

**Nearest Proximate Patch Representation (Paper 5):**
```python
class NearestProximatePatchRepresenter:
    def represent_patch(self, noisy_patch, database, blur_kernel):
        # 1. Apply same blur kernel to all database patches
        # 2. Find K nearest neighbors in blurred space
        # 3. Weighted combination in original HR space
        # Weight = exp(-||blurred_db - noisy_patch||^2 / h^2)
        ...
```

---

### Phase 5 — Colab Training (Day 3, runs overnight)

#### File: `colab_training.ipynb` [NEW] — The one notebook that does everything

The Colab notebook will have these cells in order:

```
Cell 1:  !pip install requirements + check GPU (T4/A100)
Cell 2:  Mount Google Drive, clone project repo
Cell 3:  Download CelebA (gdown) or upload small subset
Cell 4:  Train ANFIS darkness estimator (synthetic data, ~30 min)
Cell 5:  Build LCR dictionary from train set (K-means, ~20 min)
Cell 6:  Train position-patch regressors (~1 hour)
Cell 7:  Fine-tune RRDB end-to-end (use pretrained Real-ESRGAN as init, ~2 hours)
Cell 8:  Generate results on 50 test images
Cell 9:  Compute and display PSNR/SSIM/LPIPS table
Cell 10: Save all results to Drive
```

**Training strategy for 6 hours of Colab:**
- ANFIS: 200 epochs on synthetic data → fast, fully converged
- LCR dictionary: 512 atoms, K-means on 50k patches → ~20 min
- Regression: scikit-learn Ridge per position → ~1 hour on CPU
- RRDB: Load pretrained Real-ESRGAN weights → fine-tune 50 epochs → ~2 hours T4
- Total Colab time: ~5–6 hours (start overnight Day 3)

---

### Phase 6 — Backend API (Days 16–17)

#### File: `app/backend/server.py` [MODIFY]

Add these endpoints to the existing FastAPI server:

```python
POST /api/enhance          # Upload image, get enhanced result
GET  /api/metrics/{job_id} # Get computed PSNR/SSIM/LPIPS
POST /api/batch            # Batch processing endpoint
GET  /api/pipeline/info    # Return model architecture info
```

**Process flow for `/api/enhance`:**
1. Receive base64 or multipart image
2. Run `darkness_estimator.py` → compute DF
3. Run `motion_blur_handler.py` if blur detected
4. Run `anfis_lcr.py` + `regression_reconstructor.py`
5. Run `rrdb_.py` for final upsampling
6. Return enhanced image + all intermediate outputs + metrics

---

### Phase 7 — Frontend Dashboard (Days 18–20)

#### Location: `app/frontend/src/`

**Pages to build:**
1. **Upload Page** — Drag & drop, camera capture, URL input
2. **Processing Page** — Live pipeline status with animated stages
3. **Results Page** — Before/after slider, all intermediate outputs, downloadable metrics CSV
4. **Benchmark Page** — PSNR/SSIM/LPIPS charts on test set

**Key UI components:**
- `ImageComparisionSlider.tsx` — interactive before/after
- `PipelineVisualizer.tsx` — animated 4-stage pipeline diagram
- `MetricsCard.tsx` — live display of PSNR, SSIM, LPIPS, Face Acc
- `DarknessGauge.tsx` — visual indicator of the ANFIS darkness factor output

---

### Phase 8 — Evaluation & Results (Days 21–23)

**Quantitative benchmarks you must report:**

| Method | PSNR (dB) | SSIM | LPIPS | Face Acc (%) |
|---|---|---|---|---|
| Bicubic (baseline) | ~24 | ~0.72 | ~0.35 | ~45 |
| Zero-DCE only | ~26 | ~0.78 | ~0.28 | ~55 |
| ANFIS-LCR (ours, no RRDB) | target ~28 | target ~0.82 | target ~0.22 | target ~68 |
| Full ANFIS-LLFSR (ours) | target ~30 | target ~0.85 | target ~0.18 | target ~75 |

**Ablation study (critical for top grade):**
- Model without darkness estimator
- Model without LCR (replace with bicubic)
- Model without motion blur handler
- Model without regression step
- Full model

**Qualitative results:**
- 20 side-by-side comparisons (LR → ours vs. ESRGAN vs. bicubic)
- Failure case analysis (required for honest academic reporting)

---

### Phase 9 — Documentation (Days 24–25)

**What your professor will actually read:**

1. **`docs/REPORT.md`** — Full technical report (10,000 words minimum):
   - Abstract, Introduction, Literature Review (cite all 5 papers properly)
   - System Architecture with diagrams
   - Mathematical formulation of ANFIS, LCR, Ridge Regression
   - Experimental setup, Dataset, Results, Ablation Study
   - Conclusion and Future Work

2. **`docs/SETUP.md`** — Reproducible setup in 10 commands

3. **`docs/API.md`** — OpenAPI-compatible endpoint documentation

4. **Inline code documentation** — Every function must have:
   - Docstring with mathematical formula it implements
   - Reference to specific paper section number
   - Example usage

---

## File Structure (Final State)

```
BTPPF/LH/
├── app/
│   ├── backend/
│   │   ├── models/
│   │   │   ├── zero_dce.py          ✅ existing
│   │   │   ├── rrdb_generator.py    ✅ existing
│   │   │   ├── discriminator.py     ✅ existing
│   │   │   ├── arcface_model.py     ✅ existing
│   │   │   └── landmark_detector.py ✅ existing
│   │   ├── core/                    ← NEW directory
│   │   │   ├── anfis_core.py        🆕 ANFIS engine
│   │   │   ├── darkness_estimator.py 🆕 Paper 3
│   │   │   ├── anfis_lcr.py         🆕 Paper 1
│   │   │   ├── regression_reconstructor.py 🆕 Paper 4
│   │   │   └── motion_blur_handler.py 🆕 Paper 5
│   │   ├── training/
│   │   │   ├── train_anfis.py       🆕 ANFIS training
│   │   │   ├── train_sr_gan.py      🆕 Full GAN training
│   │   │   └── train_full.py        🆕 End-to-end
│   │   ├── data/
│   │   │   ├── preprocessing.py     ✅ existing
│   │   │   └── dataset.py           ✅ existing
│   │   ├── losses/
│   │   │   ├── pixel_loss.py        ✅ existing
│   │   │   ├── identity_loss.py     ✅ existing
│   │   │   └── landmark_loss.py     ✅ existing
│   │   ├── evaluation/
│   │   │   └── metrics.py           ✅ existing
│   │   ├── utils/
│   │   │   ├── model_manager.py     ✅ existing
│   │   │   └── visualization.py     ✅ existing
│   │   ├── inference.py             🔄 update to use ANFIS pipeline
│   │   └── server.py               🔄 add enhance endpoints
│   └── frontend/
│       └── src/
│           ├── components/          🆕 all UI components
│           └── pages/               🆕 all pages
├── scripts/
│   ├── download_data.py             🆕
│   └── evaluate.py                  🆕
├── docs/
│   ├── REPORT.md                    🆕 full technical report
│   ├── SETUP.md                     🆕
│   └── API.md                       🆕
├── tests/                           🆕 unit tests
│   ├── test_anfis.py
│   ├── test_darkness_estimator.py
│   └── test_lcr.py
└── requirements.txt                 🔄 add scikit-fuzzy
```

---

## Academic Differentiation Strategy

> [!NOTE]
> This is what separates a 10/10 from a 7/10 project.

### What makes this genuinely YOUR project:

1. **ANFIS from scratch in PyTorch** — Most SR papers use black-box enhancement. You implement the exact fuzzy inference mechanism described in your papers, making every parameter interpretable.

2. **Darkness Factor as a first-class signal** — DF gates every downstream stage. This is the architectural novelty not present in off-the-shelf systems.

3. **Locality Constrained Representation** — Dictionary-based face hallucination is fundamentally different from deep super-resolution. You can explain it geometrically (nearest atoms in feature space).

4. **Hybrid pipeline** — Classic signal processing (Wiener deconvolution) + handcrafted ML (Ridge regression) + deep learning (RRDB). This range demonstrates deeper understanding than pure deep learning.

5. **Ablation Study** — Shows you understand what each component contributes, not just that the system works.

---

## Examination Preparation

Your examiner will likely ask:

| Question | What to Prepare |
|---|---|
| "Explain ANFIS" | Draw the 5-layer structure on the board. Explain hybrid learning. |
| "Why locality-constrained over sparse coding?" | Computational efficiency + geometric locality = better generalization |
| "How does darkness factor improve SR?" | It gates the aggressiveness of enhancement — prevents over-brightening well-lit regions |
| "What is the closed-form LCR solution?" | `α = (D^TD + λ·diag(d²))^{-1} D^T·patch` |
| "Why combine ANFIS with deep learning?" | ANFIS provides interpretable uncertainty estimation; deep learning handles complex texture synthesis |
| "What are the limitations?" | LCR dictionary must be pre-built; ANFIS input features are hand-crafted; real-time performance |

---

## Constraints Locked In

| Constraint | Value | Impact |
|---|---|---|
| Deadline | **May 10** | 6 days total |
| GPU | **Google Colab** | T4 free tier, ~6hr sessions |
| Dataset | **CelebA standard** | 202k images, ~1.3GB, instant download |
| Training | **Overnight Colab run** | Day 3 night → results by Day 4 morning |

---

## Verification Plan

### Automated Tests
```bash
# Phase 1 — ANFIS
python -m pytest tests/test_anfis.py -v
# Verify: forward pass, hybrid learning convergence, correct DF range [0,1]

# Phase 2 — Darkness Estimator
python -m pytest tests/test_darkness_estimator.py -v
# Verify: DF ≈ 0 for bright images, DF ≈ 1 for dark images

# Phase 3 — LCR
python -m pytest tests/test_lcr.py -v
# Verify: reconstruction error < bicubic baseline

# Integration test
python scripts/evaluate.py --n_images 50 --output results/
# Produces: PSNR/SSIM table + side-by-side visuals
```

### Browser Demo Verification
- Upload a dark face image → verify all 4 pipeline stages visible
- Upload a blurred face → verify blur kernel estimated correctly
- Compare PSNR with baseline (bicubic) on same image

---

## Priority Order

If time is extremely limited, implement in this order:

1. ✅ `anfis_core.py` + `darkness_estimator.py` — Core academic contribution
2. ✅ `anfis_lcr.py` — Paper 1 direct implementation
3. ✅ Update `inference.py` to use ANFIS pipeline
4. ✅ `regression_reconstructor.py` — Paper 4
5. ✅ Backend API enhance endpoint
6. ✅ Frontend upload + results UI
7. ⏩ `motion_blur_handler.py` — Paper 5 (can scope as future work if short on time)
8. ⏩ Full GAN training loop (use pretrained RRDB if no GPU time)
9. ⏩ Full documentation

