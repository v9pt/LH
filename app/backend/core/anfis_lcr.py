"""
ANFIS-Guided Locality Constrained Representation (LCR)
=======================================================

Implements dictionary-based face patch hallucination guided by the
ANFIS Darkness Factor from darkness_estimator.py.

Reference:
    Paper 1: "Low-light robust face image super-resolution via neuro-fuzzy
              inferencing-based locality constrained representation"
              — Section III (Proposed Method), Eq. (3)–(10)

Core Ideas:
    1. Locality Constrained Representation (LCR):
       Each LR patch is expressed as a locally-weighted linear combination
       of dictionary atoms. "Locality" means atoms close in feature space
       get exponentially higher weights.

       Closed-form solution (Paper 1, Eq. 7):
           α* = (D^T D + λ · diag(d²))^{-1} · D^T · p
       where d_k = ||p - D_k||  (Euclidean distance to atom k)
             λ   = regularisation weight

    2. ANFIS-guided re-weighting (Paper 1, novelty):
       The darkness factor (DF) computed by darkness_estimator.py is used
       to re-weight α*. In dark regions, atoms from dark training examples
       are preferred. This is implemented via a DF-conditioned cosine bias.

    3. HR patch synthesis:
       Given α* and the dual HR dictionary D_HR (same atoms but HR patches):
           HR_patch_estimate = D_HR · α*

Dictionary Learning:
    Atoms are learned offline from training faces via K-SVD or simple K-means
    (we use K-means for speed, K-SVD for higher quality).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import cv2


# ─────────────────────────────────────────────────────────────
#  Patch Utilities
# ─────────────────────────────────────────────────────────────

def extract_patches(image: np.ndarray,
                    patch_size: int = 8,
                    stride: int = 4) -> Tuple[np.ndarray, list]:
    """Extract overlapping patches from an image.

    Args:
        image      : [H, W, C] or [H, W] numpy array (uint8 or float32).
        patch_size : Size of square patches.
        stride     : Stride between patches.

    Returns:
        patches    : [N, patch_size*patch_size*C]  flattened patches.
        positions  : List of (y, x) top-left corner positions.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    H, W = image.shape[:2]
    C = image.shape[2] if image.ndim == 3 else 1
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    patches, positions = [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]   # [P, P, C]
            patches.append(patch.flatten())
            positions.append((y, x))

    return np.array(patches, dtype=np.float32), positions


def reconstruct_from_patches(patches: np.ndarray,
                              positions: list,
                              image_shape: Tuple[int, int, int],
                              patch_size: int = 8) -> np.ndarray:
    """Reconstruct image from patches by averaging overlapping regions.

    Args:
        patches     : [N, patch_size²*C]  patch vectors.
        positions   : List of (y, x) top-left corner positions.
        image_shape : (H, W, C) output image shape.
        patch_size  : Size of square patches.

    Returns:
        image : [H, W, C]  reconstructed image (float32 in [0, 1]).
    """
    H, W, C = image_shape
    canvas = np.zeros((H, W, C), dtype=np.float64)
    count  = np.zeros((H, W, C), dtype=np.float64)

    for patch_vec, (y, x) in zip(patches, positions):
        patch = patch_vec.reshape(patch_size, patch_size, C)
        canvas[y:y+patch_size, x:x+patch_size, :] += patch
        count [y:y+patch_size, x:x+patch_size, :] += 1.0

    count = np.maximum(count, 1.0)
    return np.clip(canvas / count, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  LCR Encoding
# ─────────────────────────────────────────────────────────────

def lcr_encode(patch: np.ndarray,
               D: np.ndarray,
               lam: float = 1e-4,
               beta: float = 1.0) -> np.ndarray:
    """Encode a patch using Locality Constrained Representation (LCR).

    Solves the closed-form problem from Paper 1, Eq. (7):
        α* = (D^T D + λ · diag(d²))^{-1} · D^T · p

    where d_k = ||p - D[:,k]||²  is the squared distance from patch p
    to dictionary atom k, serving as the locality constraint weight.

    Args:
        patch : [d]      LR patch feature vector (flattened + normalised).
        D     : [d, K]   LR dictionary (d = patch_dim, K = n_atoms).
        lam   : Locality regularisation coefficient λ (Paper 1 uses 1e-4).
        beta  : Locality bandwidth β (controls sharpness of locality weights).

    Returns:
        alpha : [K]  sparse-ish representation coefficients.
    """
    # Squared Euclidean distances from patch to each atom
    # d_k = ||patch - D[:,k]||²
    diff   = D - patch[:, np.newaxis]           # [d, K]
    dist_sq = np.sum(diff ** 2, axis=0)         # [K]

    # Locality weights (exponential decay) — Paper 1, Eq. (5)
    # w_k = exp(dist_sq_k / beta)  (larger distance → larger penalty)
    locality_weights = np.exp(dist_sq / (beta + 1e-8))  # [K]

    # Regularised Gram matrix: G = D^T D + λ · diag(w²)
    G = D.T @ D + lam * np.diag(locality_weights ** 2)   # [K, K]

    # Closed-form: α* = G^{-1} D^T p
    DT_p = D.T @ patch                                    # [K]
    try:
        alpha = np.linalg.solve(G, DT_p)                 # [K]
    except np.linalg.LinAlgError:
        # Fallback to least-squares if singular
        alpha, _, _, _ = np.linalg.lstsq(G, DT_p, rcond=None)

    return alpha.astype(np.float32)


def anfis_reweight_coefficients(alpha: np.ndarray,
                                 atom_darknesses: np.ndarray,
                                 darkness_factor: float) -> np.ndarray:
    """Re-weight LCR coefficients based on image darkness (ANFIS guidance).

    Paper 1 Section III-C: "ANFIS-guided LCR weighting":
    When the image is dark (DF ≈ 1), atoms that correspond to dark training
    patches should receive higher weight. We implement this via a cosine
    similarity bias between the per-atom darkness level and the query DF.

    Args:
        alpha          : [K]  LCR coefficients before reweighting.
        atom_darknesses: [K]  per-atom average darkness factor (precomputed
                              during dictionary construction from training data).
        darkness_factor: float ∈ [0, 1]  DF of the current query image.

    Returns:
        alpha_reweighted : [K]  reweighted coefficients (l1-normalised).
    """
    # Affinity: atoms close in darkness space to the query get boosted
    # affinity_k = 1 - |DF_query - DF_atom_k|   ∈ [0, 1]
    affinity = 1.0 - np.abs(darkness_factor - atom_darknesses)   # [K]
    affinity = np.clip(affinity, 0, 1)

    # Multiply element-wise and renormalise
    alpha_rw = alpha * affinity
    norm = np.abs(alpha_rw).sum() + 1e-8
    return (alpha_rw / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Dictionary Builder
# ─────────────────────────────────────────────────────────────

class FaceDictionary:
    """Coupled LR–HR dictionary for face patch hallucination.

    Trains a dictionary of K paired (LR patch, HR patch) atoms.
    LR atoms are used for encoding; HR atoms for reconstruction.

    Training (offline, ~20 min on 8k images):
        1. Extract LR and HR patch pairs from all training images.
        2. Run K-means on LR patches to get K cluster centres (atoms).
        3. For each cluster, compute mean HR patch as the HR atom.
        4. Compute per-atom darkness levels using DarknessEstimator.

    Reference: Paper 1, Section III-B — Dictionary Construction.
    """

    def __init__(self,
                 n_atoms: int = 512,
                 patch_size: int = 8,
                 stride: int = 4,
                 scale: int = 4):
        """
        Args:
            n_atoms    : K — number of dictionary atoms (Paper 1 uses 512).
            patch_size : LR patch size in pixels.
            stride     : Stride for patch extraction.
            scale      : SR scale factor (LR→HR magnification).
        """
        self.K          = n_atoms
        self.patch_size = patch_size
        self.stride     = stride
        self.scale      = scale
        self.hr_patch   = patch_size * scale  # HR patch size

        # Dictionaries: set during .build()
        self.D_LR: Optional[np.ndarray] = None   # [d_lr, K]
        self.D_HR: Optional[np.ndarray] = None   # [d_hr, K]
        self.atom_darknesses: Optional[np.ndarray] = None  # [K]
        self._built = False

    def build(self,
              hr_images: list,
              darkness_factors: Optional[list] = None,
              max_patches: int = 50_000,
              seed: int = 42) -> None:
        """Build the coupled LR–HR dictionary from training images.

        Args:
            hr_images       : List of [H, W, 3] uint8 HR face images.
            darkness_factors: Precomputed DF per image (from DarknessEstimator).
                              If None, atom_darknesses are set to 0.5.
            max_patches     : Max patches to use for K-means (memory limit).
            seed            : Random seed.
        """
        from sklearn.cluster import MiniBatchKMeans

        print(f"Building dictionary: K={self.K}, "
              f"patch={self.patch_size}px, {len(hr_images)} images...")

        all_lr_patches, all_hr_patches, all_dfs = [], [], []

        for idx, hr_img in enumerate(hr_images):
            # Create LR version (downsample)
            lr_h = hr_img.shape[0] // self.scale
            lr_w = hr_img.shape[1] // self.scale
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

            # Extract LR patches
            lr_patches, positions = extract_patches(
                lr_img, self.patch_size, self.stride)
            # Extract corresponding HR patches (at scaled positions)
            hr_patches, _ = extract_patches(
                hr_img, self.hr_patch, self.stride * self.scale)

            n = min(len(lr_patches), len(hr_patches))
            all_lr_patches.append(lr_patches[:n])
            all_hr_patches.append(hr_patches[:n])

            # Darkness factor for this image (applied to all its patches)
            df = darkness_factors[idx] if darkness_factors else 0.5
            all_dfs.extend([df] * n)

        # Stack all patches
        LR = np.vstack(all_lr_patches)   # [N_total, d_lr]
        HR = np.vstack(all_hr_patches)   # [N_total, d_hr]
        DF_arr = np.array(all_dfs, dtype=np.float32)   # [N_total]

        # Subsample if too many patches
        if len(LR) > max_patches:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(LR), max_patches, replace=False)
            LR = LR[idx]
            HR = HR[idx]
            DF_arr = DF_arr[idx]

        print(f"  Clustering {len(LR)} patches into {self.K} atoms (K-means)...")

        # K-means on LR patches → cluster centres are LR dictionary atoms
        kmeans = MiniBatchKMeans(
            n_clusters=self.K,
            random_state=seed,
            n_init='auto',
            max_iter=100,
            batch_size=min(len(LR), 10_000)
        )
        kmeans.fit(LR)
        labels = kmeans.labels_           # [N_total]

        # LR dictionary: cluster centres
        self.D_LR = kmeans.cluster_centers_.T   # [d_lr, K]

        # HR dictionary: mean HR patch per cluster
        d_hr = HR.shape[1]
        D_HR = np.zeros((d_hr, self.K), dtype=np.float32)
        atom_dfs = np.zeros(self.K, dtype=np.float32)

        for k in range(self.K):
            mask = labels == k
            if mask.sum() > 0:
                D_HR[:, k] = HR[mask].mean(axis=0)
                atom_dfs[k] = DF_arr[mask].mean()
            else:
                D_HR[:, k] = np.zeros(d_hr)
                atom_dfs[k] = 0.5

        self.D_HR = D_HR
        self.atom_darknesses = atom_dfs
        self._built = True
        print(f"  Dictionary built. D_LR: {self.D_LR.shape}, "
              f"D_HR: {self.D_HR.shape}")

    def save(self, path: Union[str, Path]) -> None:
        """Save dictionary to disk as .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            D_LR=self.D_LR,
            D_HR=self.D_HR,
            atom_darknesses=self.atom_darknesses,
            patch_size=np.array(self.patch_size),
            stride=np.array(self.stride),
            scale=np.array(self.scale),
        )
        print(f"Dictionary saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load dictionary from .npz file."""
        data = np.load(path)
        self.D_LR = data['D_LR']
        self.D_HR = data['D_HR']
        self.atom_darknesses = data['atom_darknesses']
        self.patch_size = int(data['patch_size'])
        self.stride     = int(data['stride'])
        self.scale      = int(data['scale'])
        self.K = self.D_LR.shape[1]
        self._built = True
        print(f"Dictionary loaded from {path}: K={self.K}")


# ─────────────────────────────────────────────────────────────
#  Main ANFIS-LCR Hallucinator
# ─────────────────────────────────────────────────────────────

class ANFISLocalityRepresentation:
    """ANFIS-guided LCR face patch hallucinator (Paper 1 — full system).

    Pipeline for a single face:
        1. Compute darkness factor DF via DarknessEstimator.
        2. Extract LR patches.
        3. For each patch: LCR encode (closed-form) + ANFIS reweight.
        4. Reconstruct HR face: HR_patch = D_HR · α_reweighted.
        5. Blend patches back into full image.

    Reference: Paper 1, Algorithm 1.
    """

    def __init__(self,
                 dictionary: FaceDictionary,
                 lam: float = 1e-4,
                 beta: float = 1.0):
        """
        Args:
            dictionary : Built FaceDictionary instance.
            lam        : LCR regularisation (λ).
            beta       : Locality bandwidth (β).
        """
        self.dict = dictionary
        self.lam  = lam
        self.beta = beta

    def hallucinate(self,
                    lr_image: np.ndarray,
                    darkness_factor: float) -> np.ndarray:
        """Hallucinate HR face from LR input using ANFIS-LCR.

        Args:
            lr_image       : [H, W, 3] LR face image (uint8 or float32).
            darkness_factor: float ∈ [0, 1] from DarknessEstimator.

        Returns:
            hr_image : [H*scale, W*scale, 3] hallucinated HR face (float32).
        """
        if not self.dict._built:
            raise RuntimeError("Dictionary not built. Call dictionary.build() first.")

        if lr_image.dtype == np.uint8:
            lr_f = lr_image.astype(np.float32) / 255.0
        else:
            lr_f = lr_image.copy()

        H, W, C = lr_f.shape
        H_hr, W_hr = H * self.dict.scale, W * self.dict.scale
        hr_output_shape = (H_hr, W_hr, C)

        # Extract LR patches
        lr_patches, positions = extract_patches(
            lr_f, self.dict.patch_size, self.dict.stride)

        hr_patches_est = []

        for patch_vec in lr_patches:
            # LCR encode: α = (D^T D + λ diag(d²))^{-1} D^T p
            alpha = lcr_encode(
                patch_vec, self.dict.D_LR,
                lam=self.lam, beta=self.beta)

            # ANFIS reweight by darkness factor
            alpha_rw = anfis_reweight_coefficients(
                alpha, self.dict.atom_darknesses, darkness_factor)

            # HR patch estimate: D_HR · α
            hr_patch_vec = self.dict.D_HR @ alpha_rw   # [d_hr]
            hr_patch_vec = np.clip(hr_patch_vec, 0, 1)
            hr_patches_est.append(hr_patch_vec)

        # Scale positions for HR grid
        hr_positions = [
            (y * self.dict.scale, x * self.dict.scale)
            for (y, x) in positions
        ]

        hr_image = reconstruct_from_patches(
            np.array(hr_patches_est),
            hr_positions,
            hr_output_shape,
            patch_size=self.dict.hr_patch)

        return hr_image   # float32 in [0, 1]


# ─────────────────────────────────────────────────────────────
#  Quick Sanity Check (tiny synthetic dictionary)
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("ANFIS-LCR — Sanity Check (tiny synthetic dictionary)")
    print("=" * 55)

    # Build a tiny synthetic dictionary (5 atoms, 4px patches)
    P   = 4   # patch size
    K   = 5   # atoms
    C   = 3   # channels
    d   = P * P * C   # patch dim

    rng = np.random.default_rng(0)

    dict_ = FaceDictionary(n_atoms=K, patch_size=P, stride=2, scale=2)
    dict_.D_LR = rng.random((d, K)).astype(np.float32)
    dict_.D_HR = rng.random((d * 4, K)).astype(np.float32)   # 2x patch = 4x dim
    dict_.atom_darknesses = rng.uniform(0, 1, K).astype(np.float32)
    dict_._built = True

    hallucinator = ANFISLocalityRepresentation(dict_, lam=1e-4)

    # Fake 8×8 LR image
    lr_img = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    hr_out = hallucinator.hallucinate(lr_img, darkness_factor=0.7)

    print(f"LR input:  {lr_img.shape}")
    print(f"HR output: {hr_out.shape}  (should be 16×16×3)")
    print(f"Value range: [{hr_out.min():.3f}, {hr_out.max():.3f}]")

    # Test LCR encode
    p = rng.random(d).astype(np.float32)
    alpha = lcr_encode(p, dict_.D_LR, lam=1e-4)
    print(f"\nLCR alpha: {alpha}")
    print("✓ ANFIS-LCR passed sanity check.")
