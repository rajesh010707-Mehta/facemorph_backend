"""Download face_landmarker.task model if not present (runs at build time on Render)."""
import os, urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done:", MODEL_PATH)
else:
    print("Model already exists:", MODEL_PATH)
