"""
predictor.py — «одна кнопка» для предсказания эмоции.
 
Умеет работать с двумя бэкендами:
  * PyTorch-чекпоинт  checkpoints/best.pt
  * ONNX-модель       onnx/emotion.onnx   (быстрее и без torch — нужен только onnxruntime)
 
Используется в app.py, batch_test.py, evaluate.py, test_onnx.py — чтобы
предобработка картинки была ОДИНАКОВОЙ везде. Это важно: если в приложении
картинка готовится чуть иначе, чем при обучении, точность падает в пол.
"""
 
from __future__ import annotations
 
import json
from pathlib import Path
 
import cv2
import numpy as np
 
from common import BEST_PT, ONNX_MODEL, get_device
 
 
class EmotionPredictor:
    """Классификатор эмоций: картинка/кадр -> (класс, вероятность, все вероятности)."""
 
    def __init__(self,
                 model_path: str | Path | None = None,
                 onnx_path: str | Path | None = None,
                 backend: str = "auto",
                 use_clahe: bool = True):
        self.use_clahe = use_clahe
        self.device = None
        self.session = None
        self.model = None
        self.onnx_input = None
 
        if backend == "auto":
            # если ONNX уже есть — берём его (быстрее), иначе грузим PyTorch-чекпоинт
            backend = "onnx" if Path(onnx_path or ONNX_MODEL).exists() else "torch"
 
        if backend == "onnx":
            self._load_onnx(Path(onnx_path or ONNX_MODEL))
        else:
            self._load_torch(Path(model_path or BEST_PT))
 
    # ------------------------------------------------------------------ загрузка
    def _load_torch(self, path: Path) -> None:
        import torch
 
        from model import EmotionCNN
 
        if not path.exists():
            raise FileNotFoundError(f"Нет чекпоинта {path}. Сначала: python train.py")
 
        self.device = get_device()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.class_names = ckpt["class_names"]
        self.img_size = int(ckpt.get("img_size", 48))
        self.meta = {k: v for k, v in ckpt.items() if k not in {"model", "optimizer"}}
 
        self.model = EmotionCNN(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.backend = f"torch ({self.device})"
        print(f"[predictor] torch-модель загружена: {path.name} "
              f"| классов: {len(self.class_names)} | val_acc="
              f"{ckpt.get('val_acc', float('nan')):.4f}")
 
    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort
 
        if not path.exists():
            raise FileNotFoundError(f"Нет ONNX-модели {path}. Сначала: python export_onnx.py")
 
        meta_path = path.parent.parent / "checkpoints" / "classes.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.class_names = meta["class_names"]
        self.img_size = int(meta.get("img_size", 48))
        self.meta = meta
 
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.onnx_input = self.session.get_inputs()[0].name
        self.backend = f"onnx ({path.name})"
        print(f"[predictor] ONNX-модель загружена: {path.name} | классов: {len(self.class_names)}")
 
    # ------------------------------------------------------------------ подготовка
    def preprocess(self, image) -> "np.ndarray":
        """
        Любая картинка (путь, PIL, BGR-массив OpenCV) -> float32 (1, 1, H, W).
 
        Шаги: grayscale -> CLAHE (выравниваем контраст, как в FER2013)
              -> resize 48x48 -> /255 -> (x - 0.5) / 0.5
        """
        arr = self._to_bgr(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
 
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
 
        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return x.reshape(1, 1, self.img_size, self.img_size)
 
    @staticmethod
    def _to_bgr(image) -> "np.ndarray":
        """Приводит вход к BGR-массиву numpy."""
        if isinstance(image, np.ndarray):
            return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if isinstance(image, (str, Path)):
            data = np.fromfile(str(image), dtype=np.uint8)   # fromfile читает кириллицу в пути
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Не удалось прочитать изображение: {image}")
            return img
        # PIL.Image
        try:
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"Непонятный тип изображения: {type(image)} ({exc})") from exc
 
    # ------------------------------------------------------------------ предсказание
    def predict(self, image) -> dict:
        """
        Возвращает словарь:
            {'label': 'happy', 'label_ru': 'РАДОСТЬ', 'prob': 0.93,
             'index': 3, 'probs': {класс: вероятность, ...}}
        """
        x = self.preprocess(image)
 
        if self.session is not None:
            logits = self.session.run(None, {self.onnx_input: x})[0]
            logits = np.asarray(logits, dtype=np.float32)[0]
        else:
            import torch
 
            with torch.no_grad():
                t = torch.from_numpy(x).to(self.device)
                logits = self.model(t)[0].cpu().numpy()
 
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        from common import emotion_ru
 
        return {
            "label": self.class_names[idx],
            "label_ru": emotion_ru(self.class_names[idx]),
            "prob": float(probs[idx]),
            "index": idx,
            "probs": {name: float(p) for name, p in zip(self.class_names, probs)},
        }
 
 
def _softmax(logits: "np.ndarray") -> "np.ndarray":
    """Численно устойчивый softmax (вычитаем максимум, чтобы не было overflow)."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()
 
 
if __name__ == "__main__":
    # быстрая проверка: python predictor.py <путь к картинке>
    import sys
 
    path = sys.argv[1] if len(sys.argv) > 1 else None
    pred = EmotionPredictor()
    print("бэкенд:", pred.backend)
    if path:
        res = pred.predict(path)
        print(f"{res['label']} ({res['label_ru']}) prob={res['prob']:.3f}")
        print("все вероятности:", {k: round(v, 3) for k, v in res["probs"].items()})
    else:
        print("Передайте путь к картинке: python predictor.py data/fer2013/test/happy/xxx.png")
