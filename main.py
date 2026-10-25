"""
FaceMorph Match Backend — single-file version for Render deployment.
No services/ package needed.
"""
import os
import io
import urllib.request
import asyncio
import traceback
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Download model at startup ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ── Landmark extraction ───────────────────────────────────────────────────────
def extract_landmarks(image_bytes: bytes, view: str = "front"):
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to optimal size for MediaPipe (min 300px, max 1920px)
        w, h = pil_img.size
        if w < 300 or h < 300:
            scale = max(300/w, 300/h)
            pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        if w > 1920 or h > 1920:
            pil_img.thumbnail((1920, 1920), Image.LANCZOS)
        width, height = pil_img.size
        rgb_array = np.array(pil_img, dtype=np.uint8)
    except Exception:
        return None

    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    raw = result.face_landmarks[0]
    landmarks = np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in raw],
        dtype=np.float32,
    )
    return {"landmarks": landmarks, "image_width": width, "image_height": height, "view": view}

# ── Matcher ───────────────────────────────────────────────────────────────────
L_EYE, R_EYE = 33, 263
NOSE_TIP, CHIN = 1, 152
L_JAW, R_JAW   = 234, 454
L_MOUTH, R_MOUTH = 61, 291
FOREHEAD       = 10
FRONT_IDX = [L_EYE, R_EYE, NOSE_TIP, CHIN, L_JAW, R_JAW, L_MOUTH, R_MOUTH, FOREHEAD]

def _safe(lm, i):
    return lm[i] if i < len(lm) else np.zeros(3, dtype=np.float32)

def _normalize(lm):
    left  = _safe(lm, L_EYE)[:2]
    right = _safe(lm, R_EYE)[:2]
    iod   = float(np.linalg.norm(right - left)) or 1.0
    n = lm.copy(); n[:, :2] -= (left + right) / 2; n /= iod
    return n

def _key_sim(u, t, idxs):
    valid = [i for i in idxs if i < len(u) and i < len(t)]
    if not valid: return 50.0
    d = np.mean([np.linalg.norm(u[i] - t[i]) for i in valid])
    return float(max(0, 1 - d / 0.25) * 100)

def _jaw_angle(lm):
    a = _safe(lm, L_JAW)[:2]; b = _safe(lm, CHIN)[:2]; c = _safe(lm, R_JAW)[:2]
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def compute_match(user_lm, target_lm):
    u = _normalize(user_lm["front"]["landmarks"])
    t = _normalize(target_lm["front"]["landmarks"])

    front_sim = _key_sim(u, t, FRONT_IDX)

    side_sims = []
    for side in ("left", "right"):
        if side in user_lm and side in target_lm:
            us = _normalize(user_lm[side]["landmarks"])
            ts = _normalize(target_lm[side]["landmarks"])
            n  = min(len(us), len(ts))
            d  = float(np.linalg.norm(us[:n] - ts[:n], axis=1).mean())
            side_sims.append(float(max(0, 1 - d / 0.20) * 100))

    side_sim = float(np.mean(side_sims)) if side_sims else front_sim * 0.85

    jaw_u, jaw_t = _jaw_angle(u), _jaw_angle(t)
    jaw_match    = float(max(0, 1 - abs(jaw_u - jaw_t) / 30) * 100)
    nose_match   = _key_sim(u, t, list(range(1, 7)))
    profile_score = (jaw_match + side_sim) / 2
    overall = round(0.45 * front_sim + 0.35 * profile_score + 0.20 * nose_match, 2)

    insights = []
    if front_sim >= 80:  insights.append("Front facial structure matches closely.")
    elif front_sim >= 60: insights.append("Moderate front facial structure similarity detected.")
    else: insights.append("Front facial structures show notable differences.")
    if abs(jaw_u - jaw_t) < 10: insights.append(f"Very similar jaw angles ({jaw_u:.1f} vs {jaw_t:.1f} deg).")
    if nose_match >= 75: insights.append("Nose bridge and structure match well.")
    if side_sims and float(np.mean(side_sims)) >= 70: insights.append("Side profile contours align closely.")

    return {
        "overall_match_percentage": overall,
        "breakdown": {
            "front_view_similarity":   round(front_sim, 2),
            "side_profile_similarity": round(side_sim, 2),
            "jawline_angle_match":     round(jaw_match, 2),
            "nose_structure_match":    round(nose_match, 2),
        },
        "insights": insights,
    }

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="FaceMorph Match API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class MatchResult(BaseModel):
    overall_match_percentage: float
    breakdown: Dict[str, float]
    insights: List[str]

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.path.exists(MODEL_PATH)}

@app.post("/match", response_model=MatchResult)
async def match_faces(
    user_front:   UploadFile = File(...),
    target_front: UploadFile = File(...),
    user_left:    Optional[UploadFile] = File(None),
    user_right:   Optional[UploadFile] = File(None),
    target_left:  Optional[UploadFile] = File(None),
    target_right: Optional[UploadFile] = File(None),
):
    async def rd(f):
        if f is None or f.filename == "": return None
        d = await f.read(); return d or None

    try:
        uf = await rd(user_front);   tf = await rd(target_front)
        ul = await rd(user_left);    ur = await rd(user_right)
        tl = await rd(target_left);  tr = await rd(target_right)

        if not uf: raise HTTPException(400, "user_front missing")
        if not tf: raise HTTPException(400, "target_front missing")

        ulf = await asyncio.to_thread(extract_landmarks, uf, "front")
        tlf = await asyncio.to_thread(extract_landmarks, tf, "front")

        if not ulf: raise HTTPException(400, "No face detected in YOUR image. Use a clear, well-lit front-facing photo.")
        if not tlf: raise HTTPException(400, "No face detected in TARGET image. Use a clear, well-lit front-facing photo.")

        user_lm = {"front": ulf}; tgt_lm = {"front": tlf}

        for k, d, v in [("left", ul, "left"), ("right", ur, "right")]:
            if d:
                lm = await asyncio.to_thread(extract_landmarks, d, v)
                if lm: user_lm[k] = lm

        for k, d, v in [("left", tl, "left"), ("right", tr, "right")]:
            if d:
                lm = await asyncio.to_thread(extract_landmarks, d, v)
                if lm: tgt_lm[k] = lm

        return compute_match(user_lm, tgt_lm)

    except HTTPException: raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Internal error: {e}")
