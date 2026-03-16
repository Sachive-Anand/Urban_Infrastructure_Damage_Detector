# ============================================================
#  Road Damage Detector - FastAPI Backend
#  File: main.py
#  Run: uvicorn main:app --reload --port 8000
# ============================================================

import io
import os
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  App & CORS
# ─────────────────────────────────────────────
app = FastAPI(
    title="Road Damage Detector API",
    description="YOLOv8-powered road damage detection — 8 damage classes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "*"),
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Model — loaded once at startup
# ─────────────────────────────────────────────
MODEL_PATH = Path("best.pt")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Place best.pt in the same folder as main.py."
    )

log.info("Loading YOLO model from %s ...", MODEL_PATH)
model = YOLO(str(MODEL_PATH))
log.info("Model loaded. Classes: %s", model.names)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Severity map — used to add context to each detection
SEVERITY = {
    "pothole":              "High",
    "Alligator":            "High",
    "Rutting":              "High",
    "Longitudinal-Crack":   "Medium",
    "Lateral-Crack":        "Medium",
    "Edge Cracking":        "Medium",
    "Ravelling":            "Low",
    "Striping":             "Low",
}

# ─────────────────────────────────────────────
#  Response Schemas (Pydantic)
# ─────────────────────────────────────────────
class Detection(BaseModel):
    label: str
    confidence: float        # 0–100
    severity: str            # High / Medium / Low
    bbox: list[float]        # [x1, y1, x2, y2] in pixels

class PredictResponse(BaseModel):
    success: bool
    total_detections: int
    inference_time_ms: float
    image_size: dict         # {width, height}
    detections: list[Detection]
    summary: dict            # counts per label

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def validate_image(file: UploadFile, contents: bytes) -> None:
    """Raises HTTPException if file is not a valid image."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_TYPES)}"
        )
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)/1024/1024:.1f} MB). "
                   f"Max allowed: {MAX_FILE_SIZE_MB} MB."
        )


def decode_image(contents: bytes) -> np.ndarray:
    """Decodes raw bytes → OpenCV BGR array. Raises 400 on failure."""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. File may be corrupt."
        )
    return img


def run_inference(img: np.ndarray, conf: float) -> tuple:
    """Runs YOLO prediction. Returns (results, elapsed_ms)."""
    start = time.perf_counter()
    results = model.predict(img, conf=conf, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms


def build_detections(results, conf_threshold: float) -> list[Detection]:
    """Converts YOLO result boxes → list of Detection objects."""
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(
            label=label,
            confidence=round(conf * 100, 1),
            severity=SEVERITY.get(label, "Unknown"),
            bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        ))
    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def build_summary(detections: list[Detection]) -> dict:
    """Returns {label: count} dict."""
    summary = {}
    for d in detections:
        summary[d.label] = summary.get(d.label, 0) + 1
    return summary


def encode_image_png(img: np.ndarray) -> bytes:
    """Encodes OpenCV image → PNG bytes."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode result image.")
    return buffer.tobytes()

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the server is running."""
    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "classes": list(model.names.values()),
        "version": "1.0.0",
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    """Returns all detectable damage class names."""
    return {
        "classes": [
            {"id": k, "name": v, "severity": SEVERITY.get(v, "Unknown")}
            for k, v in model.names.items()
        ]
    }


@app.post("/predict/image", tags=["Predict"])
async def predict_image(
    file: UploadFile = File(..., description="Road image to analyse"),
    conf: float = Query(default=0.25, ge=0.05, le=0.95,
                        description="Confidence threshold (0.05–0.95)"),
):
    """
    Upload a road image → returns an **annotated PNG** with bounding boxes drawn.
    Use this to display the visual result in your frontend.
    """
    contents = await file.read()
    validate_image(file, contents)
    img = decode_image(contents)

    results, elapsed = run_inference(img, conf)
    annotated = results[0].plot()          # draws boxes + labels
    png_bytes = encode_image_png(annotated)

    n = len(results[0].boxes)
    log.info("predict/image | detections=%d | time=%.1fms | file=%s", n, elapsed, file.filename)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={
            "X-Inference-Time-Ms": str(round(elapsed, 1)),
            "X-Total-Detections": str(n),
        },
    )


@app.post("/predict/json", response_model=PredictResponse, tags=["Predict"])
async def predict_json(
    file: UploadFile = File(..., description="Road image to analyse"),
    conf: float = Query(default=0.25, ge=0.05, le=0.95,
                        description="Confidence threshold (0.05–0.95)"),
):
    """
    Upload a road image → returns **JSON** with all detections, confidence
    scores, bounding boxes, severity ratings, and a per-class summary.
    """
    contents = await file.read()
    validate_image(file, contents)
    img = decode_image(contents)
    h, w = img.shape[:2]

    results, elapsed = run_inference(img, conf)
    detections = build_detections(results, conf)
    summary = build_summary(detections)

    log.info("predict/json  | detections=%d | time=%.1fms | file=%s", len(detections), elapsed, file.filename)

    return PredictResponse(
        success=True,
        total_detections=len(detections),
        inference_time_ms=round(elapsed, 1),
        image_size={"width": w, "height": h},
        detections=detections,
        summary=summary,
    )


@app.post("/predict/both", tags=["Predict"])
async def predict_both(
    file: UploadFile = File(..., description="Road image to analyse"),
    conf: float = Query(default=0.25, ge=0.05, le=0.95,
                        description="Confidence threshold (0.05–0.95)"),
):
    """
    **Recommended endpoint for the React frontend.**
    Returns a JSON body containing:
    - `image_base64` — annotated PNG encoded as base64 (display directly in <img>)
    - `detections`   — full detection list with confidence + severity
    - `summary`      — per-class count
    - `inference_time_ms`
    
    One single request does everything — no need to call two endpoints.
    """
    contents = await file.read()
    validate_image(file, contents)
    img = decode_image(contents)
    h, w = img.shape[:2]

    results, elapsed = run_inference(img, conf)

    # Annotated image → base64
    annotated = results[0].plot()
    png_bytes = encode_image_png(annotated)
    import base64
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    detections = build_detections(results, conf)
    summary = build_summary(detections)

    log.info("predict/both  | detections=%d | time=%.1fms | file=%s", len(detections), elapsed, file.filename)

    return JSONResponse({
        "success": True,
        "total_detections": len(detections),
        "inference_time_ms": round(elapsed, 1),
        "image_size": {"width": w, "height": h},
        "image_base64": image_b64,          # "data:image/png;base64,..." ready for <img src>
        "detections": [d.model_dump() for d in detections],
        "summary": summary,
    })