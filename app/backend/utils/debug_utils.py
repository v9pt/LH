"""
debug_utils.py — ANFIS-LLFSR Comprehensive Debug Infrastructure
================================================================

Creates results/debug/<timestamp_run_id>/ with 9 structured sub-folders.
Writes both human-readable logging.log AND structured metrics.jsonl.

Controlled by env var ANFIS_DEBUG (default: "1" = enabled).
Set ANFIS_DEBUG=0 to suppress all debug I/O.

Sub-folders created per run:
    preprocess/   — normalised input images, BGR/RGB sanity checks
    blur_maps/    — blur score heatmaps and PSF estimates
    anfis/        — 10-D feature vectors, MF activations, DF predictions
    film/         — gamma/beta maps, blend-alpha histograms
    vq/           — VQ assignment entropy, codebook weight norms
    embeddings/   — ArcFace embedding vectors, cosine similarity matrices
    routing/      — per-image routing decisions and quality-gate verdicts
    eval/         — PSNR/SSIM/LPIPS per image, annotated visual strips
    failures/     — images that caused NaN/Inf or pipeline crashes
"""

import os
import json
import time
import logging
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import re

import numpy as np

# Optional torch import (graceful if not available during unit tests)
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Optional cv2 for image saving
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ─── Global enable/disable ──────────────────────────────────────────────────────
_DEBUG_ENABLED = os.environ.get("ANFIS_DEBUG", "1") != "0"

_SUBFOLDERS = [
    "preprocess", "blur_maps", "anfis", "film",
    "vq", "embeddings", "routing", "eval", "failures"
]


# ─── DebugLogger ────────────────────────────────────────────────────────────────

class DebugLogger:
    """
    Central debug logger for the ANFIS-LLFSR pipeline.

    Usage::

        dbg = DebugLogger()          # creates results/debug/<timestamp>/
        dbg.log("Stage 1 done")
        dbg.log_tensor(tensor, "feat_extract_output")
        dbg.log_metric("psnr", 24.5)
        dbg.save_image(arr, "preprocess/input_rgb.png")
    """

    def __init__(self,
                 base_dir: Union[str, Path] = "results/debug",
                 run_id: Optional[str] = None,
                 enabled: bool = _DEBUG_ENABLED):
        self.enabled = enabled
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / self.run_id

        if self.enabled:
            self._setup_dirs()
            self._setup_file_logger()
            self._jsonl_path = self.run_dir / "metrics.jsonl"
            # Clear JSONL for this run (dirs already fresh)
            self._jsonl_path.write_text("")
            self.info(f"DebugLogger initialised → {self.run_dir}")
        else:
            self._logger = None
            self._jsonl_path = None

        # In-memory accumulators for train/val divergence tracking
        self._train_losses: list = []
        self._val_losses: list = []

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup_dirs(self):
        """Create (or clear) the run directory and all sub-folders."""
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sf in _SUBFOLDERS:
            (self.run_dir / sf).mkdir(exist_ok=True)

    def _setup_file_logger(self):
        log_path = self.run_dir / "logging.log"
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
        datefmt = "%H:%M:%S"

        self._logger = logging.getLogger(f"anfis_debug_{self.run_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()  # avoid duplicate handlers on re-use

        # File handler (all messages)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        self._logger.addHandler(fh)

        # Console handler (INFO+)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        self._logger.addHandler(ch)

        self._logger.propagate = False

    # ── Core Logging API ───────────────────────────────────────────────────────

    def _write_jsonl(self, record: Dict[str, Any]):
        if self._jsonl_path is None:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=_json_default) + "\n")
        except Exception:
            pass

    def debug(self, msg: str):
        if self.enabled and self._logger:
            self._logger.debug(msg)

    def info(self, msg: str):
        if self.enabled and self._logger:
            self._logger.info(msg)

    def warning(self, msg: str):
        if self.enabled and self._logger:
            self._logger.warning(msg)

    def error(self, msg: str):
        if self.enabled and self._logger:
            self._logger.error(msg)

    # Alias
    log = info

    # ── Tensor Diagnostics ─────────────────────────────────────────────────────

    def log_tensor(self,
                   tensor,
                   name: str,
                   stage: str = "general") -> Dict[str, Any]:
        """Log shape, dtype, range, and NaN/Inf counts for a tensor/ndarray."""
        if not self.enabled:
            return {}

        stats = tensor_stats(tensor, name)
        msg = (
            f"[{stage}] {name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"min={stats['min']:.4f} max={stats['max']:.4f} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"nan={stats['nan_count']} inf={stats['inf_count']}"
        )
        if stats['nan_count'] > 0 or stats['inf_count'] > 0:
            self.warning(f"⚠ NaN/Inf detected — {msg}")
        else:
            self.debug(msg)

        record = {"event": "tensor_stats", "stage": stage, **stats, "ts": _ts()}
        self._write_jsonl(record)
        try:
            out = self.run_dir / "anfis" / f"{_safe_name(image_name or 'sample')}_features.npy"
            np.save(out, features.astype(np.float32))
            if activations is not None:
                np.save(
                    self.run_dir / "anfis" / f"{_safe_name(image_name or 'sample')}_mf_activations.npy",
                    np.asarray(activations, dtype=np.float32),
                )
        except Exception:
            pass
        return stats

    def check_nan_inf(self, tensor, name: str, stage: str = "general") -> bool:
        """Returns True if tensor contains NaN or Inf; logs a warning."""
        if not self.enabled:
            return False
        stats = tensor_stats(tensor, name)
        bad = stats['nan_count'] > 0 or stats['inf_count'] > 0
        if bad:
            self.warning(
                f"⚠ NaN/Inf in '{name}' at stage '{stage}': "
                f"nan={stats['nan_count']}, inf={stats['inf_count']}"
            )
            self._write_jsonl({
                "event": "nan_inf_detected", "name": name, "stage": stage,
                **stats, "ts": _ts()
            })
        return bad

    # ── Stage-specific Helpers ─────────────────────────────────────────────────

    def log_preprocess(self, img_rgb: np.ndarray, label: str = "input"):
        """Log input image statistics after loading/colour conversion."""
        if not self.enabled:
            return
        h, w = img_rgb.shape[:2]
        mean_lum = float(img_rgb.mean() / 255.0)
        self.info(f"[preprocess] {label}: H={h} W={w} mean_lum={mean_lum:.4f}")
        self._write_jsonl({
            "event": "preprocess", "label": label,
            "height": h, "width": w, "mean_lum": mean_lum,
            "dtype": str(img_rgb.dtype), "ts": _ts()
        })
        self.save_image(img_rgb, f"preprocess/{label}.png")

    def log_blur_score(self, blur_info: Dict[str, Any], image_name: str = ""):
        """Log multi-factor blur analysis results."""
        if not self.enabled:
            return
        sev = blur_info.get("blur_severity", -1.0)
        self.info(
            f"[blur] {image_name}: severity={sev:.4f} "
            f"laplacian={blur_info.get('laplacian_var', '?')} "
            f"tenengrad={blur_info.get('tenengrad', '?')} "
            f"fft={blur_info.get('fft_hf_ratio', blur_info.get('fft_score', '?'))}"
        )
        self._write_jsonl({
            "event": "blur_score", "image": image_name, **blur_info, "ts": _ts()
        })

    def log_anfis(self,
                  features: np.ndarray,
                  df_raw: float,
                  df_final: float,
                  activations: Optional[np.ndarray] = None,
                  image_name: str = ""):
        """Log ANFIS 10-D feature vector, raw DF, final DF, and MF activations."""
        if not self.enabled:
            return
        feat_str = " ".join(f"{v:.3f}" for v in features.flatten())
        self.info(
            f"[anfis] {image_name}: df_raw={df_raw:.4f} df_final={df_final:.4f} "
            f"features=[{feat_str}]"
        )
        record: Dict[str, Any] = {
            "event": "anfis_estimate",
            "image": image_name,
            "df_raw": df_raw,
            "df_final": df_final,
            "features": features.flatten().tolist(),
            "ts": _ts()
        }
        if activations is not None:
            record["mf_activations"] = activations.flatten().tolist()
            act_entropy = _entropy(activations.flatten())
            record["mf_entropy"] = act_entropy
            if act_entropy < 0.1:
                self.warning(
                    f"⚠ [anfis] Low MF entropy ({act_entropy:.4f}) — "
                    "possible class collapse."
                )
        self._write_jsonl(record)

    def log_film(self,
                 enhance_alpha: float,
                 gamma_stats: Optional[Dict] = None,
                 beta_stats: Optional[Dict] = None,
                 stage: str = "film"):
        """Log FiLM blend alpha and optional gamma/beta statistics."""
        if not self.enabled:
            return
        self.info(f"[film] enhance_alpha={enhance_alpha:.4f}")
        record: Dict[str, Any] = {
            "event": "film_modulation",
            "stage": stage,
            "enhance_alpha": enhance_alpha,
            "ts": _ts()
        }
        if gamma_stats:
            record["gamma"] = gamma_stats
            self.debug(
                f"[film] gamma: min={gamma_stats.get('min', '?'):.4f} "
                f"max={gamma_stats.get('max', '?'):.4f} "
                f"mean={gamma_stats.get('mean', '?'):.4f}"
            )
        if beta_stats:
            record["beta"] = beta_stats
        self._write_jsonl(record)

    def log_vq(self,
               assignment_entropy: float,
               codebook_norm: float,
               latent_weight: float,
               stage: str = "vq"):
        """Log VQ-codebook assignment entropy and codebook health."""
        if not self.enabled:
            return
        self.info(
            f"[vq] assignment_entropy={assignment_entropy:.4f} "
            f"codebook_norm={codebook_norm:.4f} "
            f"latent_weight={latent_weight:.4f}"
        )
        if assignment_entropy < 0.5:
            self.warning(
                f"⚠ [vq] Very low assignment entropy ({assignment_entropy:.4f}) "
                "— codebook may be collapsing to few atoms."
            )
        self._write_jsonl({
            "event": "vq_projection",
            "stage": stage,
            "assignment_entropy": assignment_entropy,
            "codebook_norm": codebook_norm,
            "latent_weight": latent_weight,
            "ts": _ts()
        })

    def log_embedding(self,
                      sim: float,
                      stage: str,
                      image_name: str = ""):
        """Log ArcFace cosine similarity between two embeddings."""
        if not self.enabled:
            return
        verdict = "PASS" if sim >= 0.90 else "FAIL"
        self.info(
            f"[embedding] {image_name} {stage}: "
            f"cosine_sim={sim:.4f} [{verdict}]"
        )
        self._write_jsonl({
            "event": "embedding_similarity",
            "stage": stage,
            "image": image_name,
            "cosine_sim": sim,
            "pass": sim >= 0.90,
            "ts": _ts()
        })

    def save_embedding(self, embedding, name: str, stage: str = "embeddings"):
        """Persist an embedding vector and log its health."""
        if not self.enabled:
            return
        try:
            if _TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                arr = embedding.detach().cpu().float().numpy()
            else:
                arr = np.asarray(embedding, dtype=np.float32)
            np.save(self.run_dir / "embeddings" / f"{_safe_name(name)}.npy", arr)
            self.log_tensor(arr, name=name, stage=stage)
        except Exception as e:
            self.warning(f"save_embedding failed for {name}: {e}")

    def log_routing(self,
                    decision: str,
                    reason: str,
                    scores: Optional[Dict] = None,
                    image_name: str = ""):
        """Log which restoration branch was chosen and why."""
        if not self.enabled:
            return
        self.info(f"[routing] {image_name}: decision='{decision}' reason='{reason}'")
        record: Dict[str, Any] = {
            "event": "routing_decision",
            "image": image_name,
            "decision": decision,
            "reason": reason,
            "ts": _ts()
        }
        if scores:
            record["scores"] = scores
        self._write_jsonl(record)

    def log_face_detection(self,
                           count: int,
                           stage: str,
                           image_name: str = ""):
        """Log how many faces were detected at a given stage."""
        if not self.enabled:
            return
        self.info(f"[face_det] {image_name} @ {stage}: {count} face(s) detected")
        self._write_jsonl({
            "event": "face_detection",
            "stage": stage,
            "image": image_name,
            "count": count,
            "ts": _ts()
        })

    def log_metrics(self,
                    psnr: Optional[float] = None,
                    ssim: Optional[float] = None,
                    lpips: Optional[float] = None,
                    identity_sim: Optional[float] = None,
                    image_name: str = ""):
        """Log PSNR / SSIM / LPIPS / identity similarity for one image."""
        if not self.enabled:
            return
        
        # Handle case where first arg might be a dict (old call style)
        if isinstance(psnr, dict):
            d = psnr
            psnr = d.get("psnr")
            ssim = d.get("ssim")
            lpips = d.get("lpips")
            identity_sim = d.get("identity_sim") or d.get("face_sim_sr")

        parts = []
        if psnr is not None:
            parts.append(f"PSNR={float(psnr):.3f}dB")
        if ssim is not None:
            parts.append(f"SSIM={float(ssim):.4f}")
        if lpips is not None:
            parts.append(f"LPIPS={float(lpips):.4f}")
        if identity_sim is not None:
            verdict = "PASS" if float(identity_sim) >= 0.90 else "FAIL"
            parts.append(f"ID={float(identity_sim):.4f}({verdict})")
        
        self.info(f"[eval] {image_name}: {' '.join(parts)}")
        self._write_jsonl({
            "event": "eval_metrics",
            "image": image_name,
            "psnr": psnr, "ssim": ssim,
            "lpips": lpips, "identity_sim": identity_sim,
            "ts": _ts()
        })

    def log_failure(self, image_name: str, stage: str, error: str):
        """Log a pipeline failure/skip."""
        if not self.enabled:
            return
        self.error(f"[FAILURE] {image_name} at stage '{stage}': {error}")
        self._write_jsonl({
            "event": "failure",
            "image": image_name,
            "stage": stage,
            "error": error,
            "ts": _ts()
        })
        try:
            out = self.run_dir / "failures" / f"{_safe_name(image_name or stage)}_{_safe_name(stage)}.txt"
            out.write_text(str(error), encoding="utf-8")
        except Exception:
            pass

    def log_train_val(self, epoch: int, train_loss: float, val_loss: float):
        """Track train/val divergence over epochs."""
        if not self.enabled:
            return
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        divergence = abs(train_loss - val_loss)
        self.debug(
            f"[train] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} divergence={divergence:.6f}"
        )
        if len(self._train_losses) > 5 and divergence > 0.1:
            self.warning(
                f"⚠ [train] Large train/val divergence at epoch {epoch}: "
                f"{divergence:.4f} (possible overfitting)"
            )
        self._write_jsonl({
            "event": "train_val",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "divergence": divergence,
            "ts": _ts()
        })

    # ── Image Saving ───────────────────────────────────────────────────────────

    def save_image(self,
                   img: np.ndarray,
                   rel_path: str,
                   is_float: bool = False):
        """
        Save a debug image.

        Args:
            img       : HxWx3 uint8 (or float32 if is_float=True, normalised to [0,1])
            rel_path  : path relative to run_dir, e.g. 'preprocess/input.png'
            is_float  : set True if img is float32 [0,1]
        """
        if not self.enabled or not _CV2_AVAILABLE:
            return
        try:
            out_path = self.run_dir / rel_path
            out_path = out_path.with_name(_safe_name(out_path.stem) + out_path.suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_arr = img
            if is_float:
                save_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            # Convert RGB→BGR for cv2
            if save_arr.ndim == 3 and save_arr.shape[2] == 3:
                save_arr = cv2.cvtColor(save_arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), save_arr)
        except Exception as e:
            self.warning(f"save_image failed for {rel_path}: {e}")

    def save_annotated_strip(self,
                              images: list,
                              labels: list,
                              metrics_text: str,
                              filename: str):
        """
        Save a horizontal strip of images with text annotations.

        Args:
            images      : list of HxWx3 uint8 numpy arrays (all same H, W)
            labels      : list of label strings (same length as images)
            metrics_text: one-line string of PSNR/SSIM etc.
            filename    : relative path inside run_dir/eval/
        """
        if not self.enabled or not _CV2_AVAILABLE:
            return
        try:
            strips = []
            for img, lbl in zip(images, labels):
                canvas = img.copy()
                # Draw label background
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(canvas, lbl, (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                strips.append(canvas)
            strip = np.hstack(strips)
            # Metrics bar
            bar = np.zeros((30, strip.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, metrics_text, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1, cv2.LINE_AA)
            final = np.vstack([strip, bar])
            out = self.run_dir / "eval" / filename
            cv2.imwrite(str(out), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        except Exception as e:
            self.warning(f"save_annotated_strip failed: {e}")

    # ── Run Summary ────────────────────────────────────────────────────────────

    def write_summary(self, summary: Dict[str, Any]):
        """Write a final JSON summary of the run."""
        if not self.enabled:
            return
        summary_path = self.run_dir / "run_summary.json"
        try:
            summary_path.write_text(
                json.dumps(summary, indent=2, default=_json_default),
                encoding="utf-8"
            )
            self.info(f"Run summary written → {summary_path}")
        except Exception as e:
            self.warning(f"write_summary failed: {e}")


# ─── Standalone Helpers ──────────────────────────────────────────────────────

def tensor_stats(tensor, name: str = "") -> Dict[str, Any]:
    """
    Compute shape, dtype, min/max/mean/std, and NaN/Inf counts.
    Works with both torch.Tensor and np.ndarray.
    """
    try:
        if _TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().float().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float64)

        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        arr_finite = arr[np.isfinite(arr)]

        return {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(tensor.dtype if _TORCH_AVAILABLE and isinstance(tensor, torch.Tensor) else arr.dtype),
            "min": float(arr_finite.min()) if arr_finite.size > 0 else float("nan"),
            "max": float(arr_finite.max()) if arr_finite.size > 0 else float("nan"),
            "mean": float(arr_finite.mean()) if arr_finite.size > 0 else float("nan"),
            "std": float(arr_finite.std()) if arr_finite.size > 0 else float("nan"),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
    except Exception as e:
        return {
            "name": name, "shape": [], "dtype": "unknown",
            "min": float("nan"), "max": float("nan"),
            "mean": float("nan"), "std": float("nan"),
            "nan_count": -1, "inf_count": -1,
            "error": str(e)
        }


def _entropy(probs: np.ndarray) -> float:
    """Compute normalised Shannon entropy of a probability-like array."""
    p = np.asarray(probs, dtype=np.float64).flatten()
    p = np.abs(p)
    total = p.sum()
    if total < 1e-12:
        return 0.0
    p = p / total
    p = p[p > 1e-12]
    n = len(p)
    if n <= 1:
        return 0.0
    raw = -float((p * np.log(p)).sum())
    return raw / np.log(n)  # normalise to [0, 1]


def _ts() -> str:
    """ISO-8601 timestamp string for JSONL records."""
    return datetime.now().isoformat(timespec="milliseconds")


def _json_default(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _safe_name(name: str) -> str:
    """Filesystem-safe stem for debug artifacts."""
    name = str(name or "sample")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:160]


# ─── Module-level convenience logger (singleton) ─────────────────────────────

_global_logger: Optional[DebugLogger] = None


def get_logger(base_dir: str = "results/debug",
               run_id: Optional[str] = None) -> DebugLogger:
    """Return (or create) the global singleton DebugLogger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DebugLogger(base_dir=base_dir, run_id=run_id)
    return _global_logger


def reset_logger(base_dir: str = "results/debug",
                 run_id: Optional[str] = None) -> DebugLogger:
    """Force-create a fresh DebugLogger (new run_id)."""
    global _global_logger
    _global_logger = DebugLogger(base_dir=base_dir, run_id=run_id)
    return _global_logger
