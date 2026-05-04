# ANFIS-LLFSR Backend API Documentation

The backend exposes a FastAPI server running by default on `http://localhost:8001`. It provides endpoints for the end-to-end inference pipeline.

## Base URL
`http://localhost:8001/api`

---

## 1. Health Check
Checks if the server is running and the ANFIS pipeline is successfully loaded in memory.

**Endpoint:** `GET /health`

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "pipeline_ready": true,
  "n_anfis_rules": 27,
  "device": "cpu",
  "stages": [
    {
      "id": 1,
      "name": "ANFIS Darkness Estimation",
      "paper": "Paper 3",
      "active": true
    }
    // ... other stages
  ]
}
```

---

## 2. Enhance Image
Upload a low-light face image to run the full 5-stage pipeline.

**Endpoint:** `POST /enhance`
**Content-Type:** `multipart/form-data`

**Request Body:**
- `file`: The image file to upload (JPEG/PNG).

**Response (`200 OK`):**
```json
{
  "status": "success",
  "processing_time_ms": 1250.45,
  "darkness_factor": 0.8542,
  "darkness_interpretation": "Extremely dark — maximum enhancement",
  "blur_severity": 0.42,
  "blur_corrected": true,
  "input_image": "iVBORw0KGgoAAAANSUhEUgAAA...", // base64 PNG
  "deblurred_image": "iVBORw0KGgoAAAANSUhEUgAAA...", // base64 PNG
  "enhanced_image": "iVBORw0KGgoAAAANSUhEUgAAA...", // base64 PNG
  "lcr_output": "iVBORw0KGgoAAAANSUhEUgAAA...", // base64 PNG
  "final_image": "iVBORw0KGgoAAAANSUhEUgAAA...", // base64 PNG
  "input_size": [128, 128],
  "output_size": [512, 512]
}
```

---

## 3. Darkness Estimator (Standalone)
Evaluate just the ANFIS Darkness Estimator (Paper 3 contribution).

**Endpoint:** `POST /darkness`
**Content-Type:** `multipart/form-data`

**Request Body:**
- `file`: The image file to evaluate.

**Response (`200 OK`):**
```json
{
  "darkness_factor": 0.7654,
  "interpretation": "Dark — strong enhancement",
  "features": {
    "mean_intensity": 0.25,
    "std_intensity": 0.15,
    "dark_channel": 0.10,
    "entropy": 4.5
  }
}
```

---

## 4. Batch Enhancement
Process multiple images in a single request. Max 10 images.

**Endpoint:** `POST /batch`
**Content-Type:** `multipart/form-data`

**Request Body:**
- `files`: A list of image files to upload.

**Response (`200 OK`):**
```json
{
  "status": "success",
  "total_time_ms": 5430.12,
  "results": [
    {
      "filename": "face1.jpg",
      "darkness_factor": 0.45,
      "final_image": "iVBORw0KGgoAAA...",
      "status": "success"
    },
    {
      "filename": "face2.jpg",
      "darkness_factor": 0.82,
      "final_image": "iVBORw0KGgoAAA...",
      "status": "success"
    }
  ]
}
```

---

## Interactive Documentation
FastAPI provides auto-generated Swagger UI documentation. Navigate to `http://localhost:8001/api/docs` in your browser to interactively test these endpoints.
