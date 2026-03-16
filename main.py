# ============================================================
#  Road Damage Detector - FastAPI Backend with AWS S3
#  File: main.py
#  Run: uvicorn main:app --reload --port 8000
# ============================================================

import io
import os
import time
import uuid
import base64
import logging
from pathlib import Path
from datetime import datetime

import cv2
import boto3
import numpy as np
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  Load .env
#  Reads your .env file and loads all values
#  into environment so os.getenv() works
# ─────────────────────────────────────────────
load_dotenv()

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
    description="YOLOv8 road damage detection with AWS S3 storage",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://urban-infrastructure-damage-detecto.vercel.app",
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
#  AWS S3 Client
#  boto3.client() opens a connection to AWS
#  using your IAM credentials
# ─────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

BUCKET = os.getenv("S3_BUCKET_NAME")

# ─────────────────────────────────────────────
#  YOLO Model — loaded once at startup
# ─────────────────────────────────────────────
MODEL_PATH = Path("best.pt")
if not MODEL_PATH.exists():
    raise FileNotFoundError("best.pt not found. Place it in the same folder as main.py.")

log.info("Loading YOLO model from %s ...", MODEL_PATH)
model = YOLO(str(MODEL_PATH))
log.info("Model loaded. Classes: %s", model.names)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB

SEVERITY = {
    "pothole":            "High",
    "Alligator":          "High",
    "Rutting":            "High",
    "Longitudinal-Crack": "Medium",
    "Lateral-Crack":      "Medium",
    "Edge Cracking":      "Medium",
    "Ravelling":          "Low",
    "Striping":           "Low",
}

# ─────────────────────────────────────────────
#  Pydantic Schemas
# ─────────────────────────────────────────────
class Detection(BaseModel):
    label:      str
    confidence: float
    severity:   str
    bbox:       list[float]

class ScanResult(BaseModel):
    scan_id:           str
    input_image_url:   str
    output_image_url:  str
    total_detections:  int
    inference_time_ms: float
    image_size:        dict
    detections:        list[Detection]
    summary:           dict
    timestamp:         str

class DeleteRequest(BaseModel):
    scan_id:    str
    input_key:  str
    output_key: str

# ─────────────────────────────────────────────
#  S3 Helpers
# ─────────────────────────────────────────────

def upload_to_s3(file_bytes: bytes, s3_key: str, content_type: str) -> str:
    """
    Uploads bytes directly to S3 — no disk involved.
    put_object() streams bytes into the bucket
    at the path defined by s3_key.

    s3_key = file path inside bucket e.g.
      inputs/2024-01-15/abc123.jpg
      outputs/2024-01-15/abc123_annotated.png
    """
    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type,
    )
    log.info("Uploaded → s3://%s/%s", BUCKET, s3_key)
    return s3_key


def generate_presigned_url(s3_key: str, expiry: int = 3600) -> str:
    """
    Creates a temporary signed URL for a private S3 file.
    URL works for `expiry` seconds (default 1 hour).
    After that it expires — nobody can access it anymore.
    This keeps images private while still letting
    React display them temporarily.
    """
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": s3_key},
        ExpiresIn=expiry,
    )


def delete_from_s3(s3_key: str) -> bool:
    """
    Deletes a single file from S3 by its key.
    delete_object() is permanent — file is gone forever.
    Returns True on success, False on failure.
    Used when the user clicks 'Remove & Re-upload'.
    """
    try:
        s3.delete_object(Bucket=BUCKET, Key=s3_key)
        log.info("Deleted → s3://%s/%s", BUCKET, s3_key)
        return True
    except ClientError as e:
        log.error("Failed to delete %s: %s", s3_key, e)
        return False

# ─────────────────────────────────────────────
#  Image Helpers
# ─────────────────────────────────────────────

def validate_image(file: UploadFile, contents: bytes):
    """Rejects non-image files and files over 10MB."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Allowed: jpg, png, webp, bmp."
        )
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({round(len(contents)/1024/1024, 1)}MB). Max 10MB."
        )


def decode_image(contents: bytes) -> np.ndarray:
    """
    Converts raw uploaded bytes → OpenCV BGR pixel array.
    np.frombuffer turns bytes into a 1D numpy array.
    cv2.imdecode turns that array into a 2D image grid.
    """
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupt.")
    return img


def encode_image_png(img: np.ndarray) -> bytes:
    """Converts OpenCV image array → PNG bytes for storage."""
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


def build_detections(results, conf_threshold: float) -> list[Detection]:
    """
    Translates raw YOLO output into clean Detection objects.
    box.cls  → class ID → class name via model.names
    box.conf → confidence score (0-1) → multiplied by 100
    box.xyxy → bounding box [x1, y1, x2, y2] in pixels
    Sorted by confidence so highest confidence appears first.
    """
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(
            label=label,
            confidence=round(conf * 100, 1),
            severity=SEVERITY.get(label, "Unknown"),
            bbox=[round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
        ))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def build_summary(detections: list[Detection]) -> dict:
    """Returns {label: count} — e.g. {'pothole': 3, 'Alligator': 1}"""
    summary = {}
    for d in detections:
        summary[d.label] = summary.get(d.label, 0) + 1
    return summary

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms server + model + S3 are ready."""
    return {
        "status":  "ok",
        "version": "2.0.0",
        "storage": "AWS S3",
        "bucket":  BUCKET,
        "region":  os.getenv("AWS_REGION"),
        "classes": list(model.names.values()),
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    """Returns all 8 detectable damage classes with severity."""
    return {
        "classes": [
            {"id": k, "name": v, "severity": SEVERITY.get(v, "Unknown")}
            for k, v in model.names.items()
        ]
    }


@app.get("/history", tags=["History"])
def get_history(limit: int = Query(default=20, le=100)):
    """
    Lists past scans from S3 outputs/ folder.
    list_objects_v2() reads the bucket like a directory.
    Sorted newest first using LastModified timestamp.
    Each item gets a fresh presigned URL valid for 1 hour.
    """
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix="outputs/",
            MaxKeys=limit,
        )
        objects = response.get("Contents", [])
        objects.sort(key=lambda x: x["LastModified"], reverse=True)

        history = []
        for obj in objects:
            key = obj["Key"]
            url = generate_presigned_url(key)
            history.append({
                "key":           key,
                "url":           url,
                "size_kb":       round(obj["Size"] / 1024, 1),
                "last_modified": obj["LastModified"].isoformat(),
            })

        return {"total": len(history), "scans": history}

    except ClientError as e:
        log.error("S3 history error: %s", e)
        raise HTTPException(status_code=500, detail="Could not fetch history from S3.")


@app.post("/scan", response_model=ScanResult, tags=["Scan"])
async def scan_image(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.05, le=0.95),
):
    """
    Main scan pipeline:
    1. Validate + decode uploaded image
    2. Generate unique scan_id with uuid4()
    3. Upload original image → S3 inputs/
    4. Run YOLO inference
    5. Draw bounding boxes on image
    6. Upload annotated image → S3 outputs/
    7. Generate presigned URLs for both images
    8. Return full result with URLs + detections
    """

    # ── 1. Read + validate ──
    contents = await file.read()
    validate_image(file, contents)
    img  = decode_image(contents)
    h, w = img.shape[:2]

    # ── 2. Unique scan ID + paths ──
    scan_id     = str(uuid.uuid4())
    timestamp   = datetime.utcnow().isoformat()
    date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
    input_key   = f"inputs/{date_prefix}/{scan_id}.jpg"
    output_key  = f"outputs/{date_prefix}/{scan_id}_annotated.png"

    # ── 3. Save original to S3 ──
    upload_to_s3(contents, input_key, file.content_type or "image/jpeg")
    log.info("scan=%s | original saved", scan_id)

    # ── 4. Run YOLO ──
    start      = time.perf_counter()
    results    = model.predict(img, conf=conf, verbose=False)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    # ── 5. Draw boxes + save annotated to S3 ──
    annotated = results[0].plot()
    png_bytes  = encode_image_png(annotated)
    upload_to_s3(png_bytes, output_key, "image/png")
    log.info("scan=%s | annotated saved | detections=%d | time=%sms",
             scan_id, len(results[0].boxes), elapsed_ms)

    # ── 6. Generate presigned URLs ──
    input_url  = generate_presigned_url(input_key)
    output_url = generate_presigned_url(output_key)

    # ── 7. Build detections ──
    detections = build_detections(results, conf)
    summary    = build_summary(detections)

    return ScanResult(
        scan_id=scan_id,
        input_image_url=input_url,
        output_image_url=output_url,
        total_detections=len(detections),
        inference_time_ms=elapsed_ms,
        image_size={"width": w, "height": h},
        detections=detections,
        summary=summary,
        timestamp=timestamp,
    )


@app.delete("/scan/{scan_id}", tags=["Scan"])
async def delete_scan(request: DeleteRequest):
    """
    Deletes both input and output images from S3
    when user clicks 'Remove & Re-upload'.

    Deletes:
      inputs/2024-01-15/abc123.jpg
      outputs/2024-01-15/abc123_annotated.png

    Frontend sends the exact keys it received
    from the /scan response so we know what to delete.
    """
    input_deleted  = delete_from_s3(request.input_key)
    output_deleted = delete_from_s3(request.output_key)

    if not input_deleted and not output_deleted:
        raise HTTPException(status_code=500, detail="Failed to delete files from S3.")

    log.info("scan=%s | deleted from S3", request.scan_id)
    return {
        "success":        True,
        "scan_id":        request.scan_id,
        "input_deleted":  input_deleted,
        "output_deleted": output_deleted,
        "message":        "Scan deleted. You can now upload a new image."
    }