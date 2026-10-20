"""Open face models from the OpenCV Zoo (YuNet detector: MIT, SFace recognizer: Apache-2.0)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = models_dir / name
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB")


def load(models_dir: Path, input_size=(320, 320)):
    det_path, rec_path = (models_dir / n for n in MODELS)
    if not (det_path.exists() and rec_path.exists()):
        raise SystemExit("Face models missing. Run: phototag models")
    detector = cv2.FaceDetectorYN.create(str(det_path), "", input_size, 0.7, 0.3, 100)
    recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
    return detector, recognizer
