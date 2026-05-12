"""
FastAPI Server — ANFIS-LLFSR System
=====================================

Exposes the 5-stage pipeline as REST endpoints.

Endpoints:
    GET  /api/health            — Server status + model info
    POST /api/enhance           — Upload image, get all stage outputs
    POST /api/darkness          — Darkness factor only (for demo)
    GET  /api/pipeline/info     — Pipeline architecture info
    POST /api/batch             — Batch process multiple images

Scale: 8× pipeline — input resized to 64×64, output is 512×512.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import base64
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Pipeline import
from inference import ANFISFaceSRPipeline

# ─── App Setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ANFIS Low-Light Face Super-Resolution API",
    description="Neuro-Fuzzy Inferencing Based System for improving dark/low-res face images.",
    version="1.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ─── Global Pipeline (loaded once at startup) ───────────────────────────
pipeline: Optional[ANFISFaceSRPipeline] = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Loading ANFIS pipeline...")
    # Force use_sota_model=True to ensure the hardened 95% accuracy weights are used
    pipeline = ANFISFaceSRPipeline(
        device='cpu',
        use_blur_correction=True,
        use_gfpgan=True,
        use_sota_model=True,
    )
    pipeline.load_pretrained(checkpoint_dir='checkpoints')
    print("Pipeline ready.")


# ─── Helper: image IO ──────────────────────────────────────────────────

def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded file bytes to uint8 RGB numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Cannot decode image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_image(img_f: np.ndarray) -> str:
    """Encode float32 [H,W,3] image to base64 PNG string."""
    img_u8 = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def encode_uint8(img_u8: np.ndarray) -> str:
    """Encode uint8 [H,W,3] image to base64 PNG string."""
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


# ─── Response Models ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    n_anfis_rules: int
    device: str
    stages: list

class EnhanceResponse(BaseModel):
    status: str
    processing_time_ms: float
    darkness_factor: float
    darkness_interpretation: str
    blur_severity: float
    blur_corrected: bool
    # Base64-encoded PNG images for each stage
    input_image: str
    deblurred_image: str
    enhanced_image: str
    lcr_output: str
    final_image: str
    # Metadata
    input_size: list   # [H, W] of the LR input fed to the pipeline
    output_size: list  # [H, W] of the final SR output
    scale_factor: int  # upscaling factor (8 for 64→512)

class DarknessResponse(BaseModel):
    darkness_factor: float
    interpretation: str
    features: dict

class BatchEnhanceResponse(BaseModel):
    status: str
    results: list
    total_time_ms: float


# ─── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check — confirms pipeline is loaded."""
    global pipeline
    if pipeline is None:
        return HealthResponse(
            status="initialising",
            pipeline_ready=False,
            n_anfis_rules=0,
            device="cpu",
            stages=[],
        )
    info = pipeline.get_pipeline_info()
    return HealthResponse(
        status="ok",
        pipeline_ready=True,
        n_anfis_rules=info['n_anfis_rules'],
        device=info['device'],
        stages=info['stages'],
    )


@app.post("/api/enhance", response_model=EnhanceResponse)
async def enhance(file: UploadFile = File(...)):
    """Upload a low-light face image and get all pipeline stage outputs.

    Returns base64-encoded PNG images for:
        - Input (original)
        - After motion blur correction (Stage 2)
        - After Zero-DCE enhancement (Stage 3a)
        - After ANFIS-LCR hallucination (Stage 3b)
        - Final output (after RRDB, Stage 5)

    Also returns the ANFIS Darkness Factor (DF) and blur metadata.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    file_bytes = await file.read()
    img_rgb    = decode_upload(file_bytes)

    t_start = time.perf_counter()
    try:
        # 8× pipeline: target_size=512 → LR resized to 64×64, output=512×512
        results = pipeline.enhance(img_rgb, target_size=512)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Synchronized with ANFIS 2.0 Mathematical Boundaries
    df = results['darkness_factor']
    if df < 0.25:
        interp = "Well-lit — Identity preservation prioritized"
    elif df < 0.55:
        interp = "Moderate — Balanced enhancement"
    elif df < 0.80:
        interp = "Dark — Generative reconstruction active"
    else:
        interp = "Extremely dark — Maximum LCR Codebook projection"

    blur_info = results.get('blur_info', {})

    # Encode all stage images to base64
    input_b64     = encode_uint8(results['input'])
    deblurred_b64 = encode_image(results.get('deblurred', results['input'].astype(np.float32)/255))
    enhanced_b64  = encode_image(results.get('enhanced', results['input'].astype(np.float32)/255))
    lcr_b64       = encode_image(results.get('lcr_output', results['input'].astype(np.float32)/255))
    final_b64     = encode_uint8(results['final_uint8'])

    lr_h, lr_w = results['input'].shape[:2]
    hr_h, hr_w = results['final_uint8'].shape[:2]

    return EnhanceResponse(
        status="success",
        processing_time_ms=round(elapsed_ms, 2),
        darkness_factor=round(float(df), 4),
        darkness_interpretation=interp,
        blur_severity=round(float(blur_info.get('blur_severity', 0)), 4),
        blur_corrected=bool(blur_info.get('corrected', False)),
        input_image=input_b64,
        deblurred_image=deblurred_b64,
        enhanced_image=enhanced_b64,
        lcr_output=lcr_b64,
        final_image=final_b64,
        input_size=[lr_h, lr_w],
        output_size=[hr_h, hr_w],
        scale_factor=hr_h // lr_h if lr_h > 0 else 8,
    )


@app.post("/api/darkness", response_model=DarknessResponse)
async def estimate_darkness(file: UploadFile = File(...)):
    """Estimate the Darkness Factor of an uploaded image using ANFIS.

    Returns DF ∈ [0, 1], an interpretation, and the 4 illumination features.
    Useful for demonstrating Paper 3 contribution in isolation.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    file_bytes = await file.read()
    img_rgb    = decode_upload(file_bytes)

    if not pipeline.darkness_estimator._trained:
        # Heuristic fallback
        df     = 1.0 - float(img_rgb.mean() / 255.0)
        feats  = {}
        interp = "ANFIS not trained — heuristic estimate"
    else:
        desc   = pipeline.darkness_estimator.describe_prediction(img_rgb)
        df     = desc['darkness_factor']
        feats  = {k: round(v, 4) for k, v in desc['features'].items()}
        interp = desc['interpretation']

    return DarknessResponse(
        darkness_factor=round(float(df), 4),
        interpretation=interp,
        features=feats,
    )


@app.get("/api/pipeline/info")
async def pipeline_info():
    """Return complete pipeline architecture information."""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    return pipeline.get_pipeline_info()


@app.post("/api/batch")
async def batch_enhance(files: List[UploadFile] = File(...)):
    """Process multiple images in batch.

    Returns a list of results (one per image) with final image and DF.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    if len(files) > 10:
        raise HTTPException(status_code=422, detail="Max 10 images per batch.")

    t_start = time.perf_counter()
    batch_results = []

    for f in files:
        try:
            fb     = await f.read()
            img    = decode_upload(fb)
            res    = pipeline.enhance(img, target_size=128)
            batch_results.append({
                "filename": f.filename,
                "darkness_factor": round(float(res['darkness_factor']), 4),
                "final_image": encode_uint8(res['final_uint8']),
                "status": "success",
            })
        except Exception as e:
            batch_results.append({
                "filename": f.filename,
                "status": "error",
                "error": str(e),
            })

    total_ms = (time.perf_counter() - t_start) * 1000
    return BatchEnhanceResponse(
        status="success",
        results=batch_results,
        total_time_ms=round(total_ms, 2),
    )


# ─── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)