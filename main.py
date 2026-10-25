from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import traceback
import os
import urllib.request

# ── Download model at startup if missing ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading face_landmarker.task to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")
else:
    print(f"Model found at {MODEL_PATH}")

from services.landmark_extractor import extract_landmarks
from services.matcher import compute_match

app = FastAPI(title="FaceMorph Match API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchResult(BaseModel):
    overall_match_percentage: float
    breakdown: Dict[str, float]
    insights: List[str]


@app.post("/match", response_model=MatchResult)
async def match_faces(
    user_front: UploadFile = File(...),
    target_front: UploadFile = File(...),
    user_left: Optional[UploadFile] = File(None),
    user_right: Optional[UploadFile] = File(None),
    target_left: Optional[UploadFile] = File(None),
    target_right: Optional[UploadFile] = File(None),
):
    async def safe_read(upload: Optional[UploadFile]) -> Optional[bytes]:
        if upload is None or upload.filename == "":
            return None
        data = await upload.read()
        return data if data else None

    try:
        user_front_bytes   = await safe_read(user_front)
        target_front_bytes = await safe_read(target_front)
        user_left_bytes    = await safe_read(user_left)
        user_right_bytes   = await safe_read(user_right)
        target_left_bytes  = await safe_read(target_left)
        target_right_bytes = await safe_read(target_right)

        if not user_front_bytes:
            raise HTTPException(status_code=400, detail="user_front image is empty or missing")
        if not target_front_bytes:
            raise HTTPException(status_code=400, detail="target_front image is empty or missing")

        ulf = await asyncio.to_thread(extract_landmarks, user_front_bytes,   "front")
        tlf = await asyncio.to_thread(extract_landmarks, target_front_bytes, "front")

        if ulf is None:
            raise HTTPException(status_code=400,
                detail="No face detected in YOUR front image. Please use a clear, well-lit photo with the face fully visible.")
        if tlf is None:
            raise HTTPException(status_code=400,
                detail="No face detected in TARGET front image. Please use a clear, well-lit photo with the face fully visible.")

        user_lm   = {"front": ulf}
        target_lm = {"front": tlf}

        for key, data, view in [("left", user_left_bytes, "left"), ("right", user_right_bytes, "right")]:
            if data:
                lm = await asyncio.to_thread(extract_landmarks, data, view)
                if lm:
                    user_lm[key] = lm

        for key, data, view in [("left", target_left_bytes, "left"), ("right", target_right_bytes, "right")]:
            if data:
                lm = await asyncio.to_thread(extract_landmarks, data, view)
                if lm:
                    target_lm[key] = lm

        result = compute_match(user_lm, target_lm)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.path.exists(MODEL_PATH)}
