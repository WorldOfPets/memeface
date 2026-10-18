import base64
import numpy as np
import cv2
from collections import Counter, deque
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Импортируем ваши модули
from face_detector import build_detector, crop_face, expand_box

try:
    from predictor import EmotionPredictor
except ImportError:
    class EmotionPredictor:
        def __init__(self):
            self.class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            self.img_size = 48
        def predict(self, face_img: np.ndarray) -> dict:
            import random
            label = random.choice(self.class_names)
            probs = {c: 0.05 for c in self.class_names}
            probs[label] = 0.65
            return {"probs": probs, "prob": 0.65, "label": label}

# ==============================================================================
# КОНФИГУРАЦИЯ И УТИЛИТЫ
# ==============================================================================
MEMES_DIR = Path("memes")
EMOTION_COLORS = {
    "happy": (0, 255, 0), "sad": (255, 0, 0), "angry": (0, 0, 255),
    "surprise": (255, 255, 0), "neutral": (200, 200, 200), 
    "fear": (128, 0, 128), "disgust": (0, 128, 0)
}

class MemeBank:
    def __init__(self, folder: Path):
        self.memes = {}
        if not folder.exists():
            print(f"[!] Папка {folder} не найдена.")
            return
        for f in folder.iterdir():
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None: continue
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.memes[f.stem.lower()] = img
        print(f"[*] Загружено мемов: {len(self.memes)}")

    def get(self, label: str):
        return self.memes.get(label.lower(), self.memes.get("neutral", None))

    @staticmethod
    def paste(frame: np.ndarray, meme_img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        if meme_img is None or w <= 0 or h <= 0: return frame
        H, W = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        
        resized = cv2.resize(meme_img, (w, h), interpolation=cv2.INTER_AREA)
        sub = resized[y1 - y:y2 - y, x1 - x:x2 - x]
        
        bgr = sub[..., :3].astype(np.float32)
        alpha = (sub[..., 3:4].astype(np.float32) / 255.0)
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        
        frame[y1:y2, x1:x2] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        return frame

class Smoother:
    def __init__(self, class_names: list[str], window: int = 5):
        self.class_names = class_names
        self.history = deque(maxlen=window)
        self.current_probs = {c: 0.0 for c in class_names}

    def push(self, probs: dict[str, float]) -> tuple[str, dict]:
        # Плавное сглаживание вероятностей (EMA)
        alpha = 0.3
        for c in self.class_names:
            self.current_probs[c] = alpha * probs.get(c, 0.0) + (1 - alpha) * self.current_probs.get(c, 0.0)
        
        current_label = max(self.current_probs, key=self.current_probs.get)
        self.history.append(current_label)
        stable_label = Counter(self.history).most_common(1)[0][0]
        
        return stable_label, self.current_probs

# ==============================================================================
# FASTAPI ПРИЛОЖЕНИЕ
# ==============================================================================
app = FastAPI(title="MemeFace Web")

detector = None
predictor = None
memes = None
smoother = None

@app.on_event("startup")
def startup_event():
    global detector, predictor, memes, smoother
    detector = build_detector(kind="mediapipe", min_size=70)
    print(f"[*] Детектор: {detector.name}")
    
    predictor = EmotionPredictor()
    smoother = Smoother(predictor.class_names, window=5)
    memes = MemeBank(MEMES_DIR)
    print("[*] Бэкенд готов.")

# Отдаем HTML файл
@app.get("/")
async def index():
    return FileResponse("templates/MemeFace.html")

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    # Зеркалируем кадр
    frame = cv2.flip(frame, 1)

    boxes = detector.detect(frame)
    label = "neutral"
    probs = smoother.current_probs

    if boxes:
        box = boxes[0]
        face_crop = crop_face(frame, box, size=predictor.img_size, expand=1.25)
        res = predictor.predict(face_crop)
        
        label, probs = smoother.push(res["probs"])
        
        ex, ey, ew, eh = expand_box(box, frame.shape, k=1.25)
        meme_img = memes.get(label)
        frame = MemeBank.paste(frame, meme_img, ex, ey, ew, eh)
        
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        cv2.putText(frame, f"{label.upper()}", (ex, ey - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Кодируем в base64 для отправки в JSON
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{img_base64}",
        "label": label,
        "probs": probs
    })

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" позволяет открывать с телефона в той же Wi-Fi сети
    uvicorn.run(app, host="0.0.0.0", port=8000)