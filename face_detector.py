"""
face_detector.py — ищем лицо на кадре. Три варианта на выбор.
 
  haar      — каскад Хаара из OpenCV. Работает «из коробки», ничего качать не нужно.
              Минусы: теряет лицо при повороте/плохом свете, любит ложные срабатывания.
  yunet     — YuNet (OpenCV Zoo). Маленькая нейросеть 345 КБ, качается один раз
              автоматически. Заметно стабильнее Хаара. РЕКОМЕНДУЕМ.
  mediapipe — Google MediaPipe. Самый стабильный, но требует `uv pip install mediapipe`
              (работает на Python 3.9-3.12).
 
Во всех случаях на выходе — список прямоугольников (x, y, w, h), от крупного к мелкому.
"""
 
from __future__ import annotations
 
import urllib.request
from pathlib import Path
 
import cv2
import numpy as np
 
ASSETS = Path(__file__).resolve().parent / "assets"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_FILE = ASSETS / "face_detection_yunet_2023mar.onnx"
 
 
class HaarDetector:
    """Классика: каскад Хаара. Нулевая настройка, работает везде."""
 
    name = "haar"
 
    def __init__(self, min_size: int = 80, scale_factor: float = 1.1,
                 min_neighbors: int = 5):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise FileNotFoundError(f"Не найден каскад: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        self.min_size = (min_size, min_size)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
 
    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)          # выравниваем гистограмму — Хаар это любит
        rects = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=self.min_size, flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(rects) == 0:
            return []
        boxes = [tuple(int(v) for v in r) for r in rects]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
 
 
class YuNetDetector:
    """YuNet из OpenCV Zoo: tiny-нейросеть, работает быстро и точно."""
 
    name = "yunet"
 
    def __init__(self, min_size: int = 60, score_threshold: float = 0.7,
                 nms_threshold: float = 0.3, top_k: int = 20, auto_download: bool = True):
        if not YUNET_FILE.exists():
            if not auto_download:
                raise FileNotFoundError(f"Нет файла {YUNET_FILE}")
            self._download()
        # второй аргумент — пустая строка: модель уже на диске
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_FILE), "", (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self.min_size = min_size
 
    @staticmethod
    def _download() -> None:
        ASSETS.mkdir(parents=True, exist_ok=True)
        print(f"[yunet] качаю модель {YUNET_FILE.name} (~345 КБ) ...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_FILE)
        print("[yunet] готово")
 
    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _ret, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return []
        boxes = []
        for f in faces:
            x, y, bw, bh = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
            if bw < self.min_size or bh < self.min_size:
                continue
            x = max(0, x); y = max(0, y)
            bw = min(bw, w - x); bh = min(bh, h - y)
            boxes.append((x, y, bw, bh))
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
 
 
class MediaPipeDetector:
    """MediaPipe Face Detection (новый Tasks API, версия 0.10.8+)."""

    name = "mediapipe"
    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                 "face_detector/blaze_face_short_range/float16/latest/"
                 "blaze_face_short_range.tflite")
    MODEL_FILE = ASSETS / "blaze_face_short_range.tflite"

    def __init__(self, min_confidence: float = 0.5, min_size: int = 60,
                 auto_download: bool = True):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector, FaceDetectorOptions, RunningMode
        )
        import mediapipe as mp

        self.mp = mp
        self.min_size = min_size

        if not self.MODEL_FILE.exists():
            if not auto_download:
                raise FileNotFoundError(f"Нет файла {self.MODEL_FILE}")
            self._download()

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(self.MODEL_FILE)),
            min_detection_confidence=min_confidence,
        )
        self.detector = FaceDetector.create_from_options(options)

    @staticmethod
    def _download() -> None:
        ASSETS.mkdir(parents=True, exist_ok=True)
        print(f"[mediapipe] качаю модель {MediaPipeDetector.MODEL_FILE.name} ...")
        urllib.request.urlretrieve(MediaPipeDetector.MODEL_URL,
                                   MediaPipeDetector.MODEL_FILE)
        print("[mediapipe] готово")

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Новый API требует mp.Image
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.detections:
            return []

        boxes = []
        for det in result.detections:
            bb = det.bounding_box
            x = max(0, int(bb.origin_x))
            y = max(0, int(bb.origin_y))
            bw = min(w - x, int(bb.width))
            bh = min(h - y, int(bb.height))
            if bw < self.min_size or bh < self.min_size:
                continue
            boxes.append((x, y, bw, bh))

        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def close(self) -> None:
        self.detector.close()
 
 
def build_detector(kind: str = "yunet", min_size: int = 70, fallback: bool = True):
    """
    Фабрика детекторов. Если запрошенный не завёлся — тихо падаем на haar,
    чтобы приложение работало всегда.
    """
    kind = (kind or "yunet").lower()
    if kind == "haar":
        return HaarDetector(min_size=min_size)
    if kind == "yunet":
        try:
            return YuNetDetector(min_size=min_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[detector] YuNet не завёлся ({exc})")
            if not fallback:
                raise
            print("[detector] падаем на haar")
            return HaarDetector(min_size=min_size)
    if kind == "mediapipe":
        try:
            return MediaPipeDetector(min_size=min_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[detector] MediaPipe не завёлся ({exc})")
            if not fallback:
                raise
            print("[detector] падаем на haar")
            return HaarDetector(min_size=min_size)
    raise ValueError(f"Неизвестный детектор: {kind} (haar | yunet | mediapipe)")
 
 
def expand_box(box: tuple[int, int, int, int], frame_shape, k: float = 1.25):
    """
    Расширяем бокс в k раз от центра.
 
    Зачем: FER2013 обучен на кропах, где видно лоб, щёки и подбородок, а детектор
    выдаёт рамку впритык к лицу. Без расширения точность на веб-камере падает.
    """
    x, y, w, h = box
    H, W = frame_shape[:2]
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * k, h * k
    x1 = int(max(0, cx - nw / 2)); y1 = int(max(0, cy - nh / 2))
    x2 = int(min(W, cx + nw / 2)); y2 = int(min(H, cy + nh / 2))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)
 
 
def crop_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int],
              size: int = 48, expand: float = 1.25) -> np.ndarray:
    """Вырезает лицо и приводит к квадратному размеру size x size (BGR)."""
    x, y, w, h = expand_box(box, frame_bgr.shape, expand)
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
 
 
if __name__ == "__main__":
    # быстрая проверка: python face_detector.py [путь к картинке]
    import sys
 
    det = build_detector("yunet")
    print("Детектор:", det.name)
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        print("Найдено лиц:", len(det.detect(img)))
    else:
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if ok:
            boxes = det.detect(frame)
            print(f"Кадр с веб-камеры {frame.shape[1]}x{frame.shape[0]}, лиц найдено: {len(boxes)}")
            for b in boxes:
                print("   box:", b)
        else:
            print("Веб-камера не открылась")
