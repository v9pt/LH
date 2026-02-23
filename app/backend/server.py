"""FastAPI server for Face Hallucination System."""

from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Import face hallucination pipeline
try:
    from inference import FaceHallucinationPipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import pipeline: {e}")
    PIPELINE_AVAILABLE = False


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
if mongo_url:
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'face_hallucination')]
else:
    client = None
    db = None
    print("Warning: MongoDB not configured")

# FastAPI app
app = FastAPI(title="Face Hallucination API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Lazy-loaded pipeline
pipeline = None


def get_pipeline():
    """Get or initialize the face hallucination pipeline."""
    global pipeline

    if pipeline is None:
        if not PIPELINE_AVAILABLE:
            return None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing Face Hallucination Pipeline on {device}...")

        pipeline = FaceHallucinationPipeline(device=device, use_enhancer=True)

        try:
            pipeline.load_pretrained_models()
        except Exception as e:
            print(f"Could not load pretrained models: {e}")
            print("Using randomly initialized weights (for demo)")

    return pipeline


# =====================
# Pydantic Models
# =====================
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatusCheckCreate(BaseModel):
    client_name: str


class EnhancementResult(BaseModel):
    success: bool
    message: str
    sr_image_base64: Optional[str] = None
    enhanced_image_base64: Optional[str] = None
    input_dimensions: Optional[dict] = None
    output_dimensions: Optional[dict] = None


class ModelInfo(BaseModel):
    device: str
    enhancer_enabled: bool
    generator_type: str
    upscale_factor: str
    status: str


# =====================
# Routes
# =====================
@api_router.get("/")
async def root():
    return {
        "message": "Face Hallucination API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "model_info": "/api/model-info",
            "enhance": "/api/enhance"
        }
    }


@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_available": PIPELINE_AVAILABLE,
        "pipeline_initialized": pipeline is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mongodb_connected": db is not None
    }


@api_router.get("/model-info", response_model=ModelInfo)
async def model_info():
    pipe = get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not available")

    info = pipe.get_model_info()
    info['status'] = 'ready'
    return info


@api_router.post("/enhance", response_model=EnhancementResult)
async def enhance_image(file: UploadFile = File(...)):
    pipe = get_pipeline()
    if pipe is None:
        return EnhancementResult(success=False, message="Pipeline unavailable")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        results = pipe.enhance_and_super_resolve(image)
        sr_image = results['sr_image']
        enhanced_image = results['enhanced_image']

        oh, ow = sr_image.shape[:2]

        def to_base64(img):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()

        return EnhancementResult(
            success=True,
            message="Image enhanced successfully",
            sr_image_base64=to_base64(sr_image),
            enhanced_image_base64=to_base64(enhanced_image),
            input_dimensions={"width": w, "height": h},
            output_dimensions={"width": ow, "height": oh}
        )

    except Exception as e:
        logging.exception("Enhancement failed")
        return EnhancementResult(success=False, message=str(e))


@api_router.post("/enhance-binary")
async def enhance_image_binary(file: UploadFile = File(...)):
    pipe = get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline unavailable")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sr_image = pipe.enhance_and_super_resolve(image)['sr_image']

    buf = io.BytesIO()
    Image.fromarray(sr_image).save(buf, format='PNG')
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


# MongoDB demo endpoints
if db is not None:
    @api_router.post("/status", response_model=StatusCheck)
    async def create_status_check(input: StatusCheckCreate):
        status = StatusCheck(**input.dict())
        await db.status_checks.insert_one(status.dict())
        return status

    @api_router.get("/status", response_model=List[StatusCheck])
    async def get_status_checks():
        docs = await db.status_checks.find().to_list(1000)
        return [StatusCheck(**d) for d in docs]


# =====================
# App setup
# =====================
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@app.on_event("startup")
async def startup_event():
    logging.info("Face Hallucination API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()
    logging.info("Shutting down")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)