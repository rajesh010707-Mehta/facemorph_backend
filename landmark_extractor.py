"""
landmark_extractor.py
Extract 478 3-D facial landmarks using MediaPipe FaceLandmarker (Tasks API).
Compatible with mediapipe >= 0.10 / 1.x.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Optional, Dict, Any

# Path to the .task model bundled with the backend
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
_MODEL_PATH = os.path.abspath(_MODEL_PATH)


def _get_landmarker() -> mp_vision.FaceLandmarker:
    """Create a FaceLandmarker for single-image (non-live) use."""
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_landmarks(image_bytes: bytes, view: str = "front") -> Optional[Dict[str, Any]]:
    """
    Extract 3-D facial landmarks from raw image bytes.

    Args:
        image_bytes: Raw binary image data (JPEG / PNG / WEBP …).
        view: Informational tag – 'front', 'left', or 'right'.

    Returns:
        dict with keys:
            landmarks  : np.ndarray (N, 3)  –  (x_px, y_px, z_relative)
            image_width: int
            image_height: int
            view       : str
        None if no face is detected or the image cannot be decoded.
    """
    # Decode bytes → OpenCV BGR
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    height, width = bgr.shape[:2]

    # MediaPipe Tasks API expects an mp.Image (RGB)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with _get_landmarker() as landmarker:
        detection_result = landmarker.detect(mp_image)

    if not detection_result.face_landmarks:
        return None

    # First (and only) face
    raw = detection_result.face_landmarks[0]

    # Convert normalised coords → absolute pixel space
    # z in MediaPipe is relative to image width (same scale as x)
    landmarks = np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in raw],
        dtype=np.float32,
    )  # shape (478, 3)  — Tasks API returns 478 pts (incl. iris)

    return {
        "landmarks": landmarks,
        "image_width": width,
        "image_height": height,
        "view": view,
    }
