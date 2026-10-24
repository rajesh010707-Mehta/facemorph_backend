"""
landmark_extractor.py
Extract 478 3-D facial landmarks using MediaPipe FaceLandmarker (Tasks API).
Uses PIL as primary decoder to avoid libGL dependency on headless servers.
"""

import os
import io
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from typing import Optional, Dict, Any

# Absolute path to model file
_MODEL_PATH = r"d:\scan1\Face matching\backend\face_landmarker.task"

# Fallback paths for different environments
if not os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_landmarker.task")
    )
if not os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    )


def extract_landmarks(image_bytes: bytes, view: str = "front") -> Optional[Dict[str, Any]]:
    """
    Extract 3-D facial landmarks from raw image bytes.
    Uses PIL for decoding (no libGL required).

    Args:
        image_bytes: Raw binary image data (JPEG / PNG / WEBP).
        view: 'front', 'left', or 'right'.

    Returns:
        dict with landmarks (np.ndarray Nx3), image_width, image_height, view
        None if no face detected or image cannot be decoded.
    """
    try:
        # Use PIL to decode — no libGL needed
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = pil_img.size
        rgb_array = np.array(pil_img, dtype=np.uint8)
    except Exception:
        return None

    # Create MediaPipe Image from numpy RGB array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)

    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        detection_result = landmarker.detect(mp_image)

    if not detection_result.face_landmarks:
        return None

    raw = detection_result.face_landmarks[0]
    landmarks = np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in raw],
        dtype=np.float32,
    )

    return {
        "landmarks": landmarks,
        "image_width": width,
        "image_height": height,
        "view": view,
    }
