"""
common.py — общие настройки проекта MemeFace.

Здесь живёт всё, что нужно другим скриптам: пути к папкам, выбор устройства
(CPU/GPU), список эмоций и функции сохранения/загрузки метаданных модели.

Идея простая: никакие пути и названия классов НЕ хардкодим в десяти местах,
а держим в одном файле. Поменяли папку — поменяли одну строчку.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Пути проекта (всё относительно этого файла)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"          # сюда качаем «сырой» датасет FER2013
DATA = ROOT / "data" / "fer2013"          # подготовленный датасет: train/val/test
RUNS = ROOT / "runs"                      # результаты обучения
CKPT_DIR = ROOT / "checkpoints"           # best.pt, last.pt, classes.json, history.json
ONNX_DIR = ROOT / "onnx"                  # emotion.onnx
MEMES_DIR = ROOT / "memes"                # angry.png, happy.png, ...
REPORTS = ROOT / "reports"                # графики, матрица ошибок, отчёты

for _p in (DATA_RAW, DATA, RUNS, CKPT_DIR, ONNX_DIR, MEMES_DIR, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

CLASSES_JSON = CKPT_DIR / "classes.json"   # метаданные обученной модели
BEST_PT = CKPT_DIR / "best.pt"             # лучшая модель (веса + метаданные)
LAST_PT = CKPT_DIR / "last.pt"             # модель последней эпохи
HISTORY_JSON = CKPT_DIR / "history.json"   # история обучения (loss/acc по эпохам)
ONNX_MODEL = ONNX_DIR / "emotion.onnx"

# ---------------------------------------------------------------------------
# 2. Гиперпараметры модели и данных
# ---------------------------------------------------------------------------
IMG_SIZE = 48          # FER2013 — это 48x48, больше не нужно (и дольше считать)
NUM_CHANNELS = 1       # изображения чёрно-белые (grayscale)
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
SEED = 42

# Порядок классов в ОРИГИНАЛЬНОМ fer2013.csv (0..6).
# ВНИМАНИЕ: torchvision.datasets.ImageFolder сортирует папки ПО АЛФАВИТУ,
# поэтому порядок будет другой: angry, disgust, fear, happy, NEUTRAL, sad, surprise.
# Мы всегда читаем порядок из classes.json, который сохраняется при обучении.
FER2013_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Человеческие названия для интерфейса (ключ — имя папки/класса)
EMOTIONS_RU = {
    "angry": "ЗЛОСТЬ",
    "disgust": "ОТВРАЩЕНИЕ",
    "fear": "СТРАХ",
    "happy": "РАДОСТЬ",
    "sad": "ГРУСТЬ",
    "surprise": "УДИВЛЕНИЕ",
    "neutral": "НЕЙТРАЛЬНО",
    "sleepy": "СОННЫЙ",
    "focused": "ФОКУС",
    "meme": "МЕМ",
}

# Цвета эмоций (BGR — так рисует OpenCV) для рамок и текста в приложении
EMOTION_COLORS = {
    "angry": (60, 60, 235),
    "disgust": (90, 140, 90),
    "fear": (170, 90, 200),
    "happy": (40, 200, 255),
    "sad": (200, 150, 60),
    "surprise": (60, 220, 220),
    "neutral": (190, 190, 190),
}


def emotion_ru(name: str) -> str:
    """Русское название эмоции; если класса нет в словаре — вернём как есть."""
    return EMOTIONS_RU.get(name, name.upper())


def emotion_color(name: str):
    """Цвет для эмоции (BGR). Неизвестные классы рисуем белым."""
    return EMOTION_COLORS.get(name, (255, 255, 255))


# ---------------------------------------------------------------------------
# 3. Устройство: GPU, если он есть, иначе CPU
# ---------------------------------------------------------------------------
def get_device():
    """Возвращает torch.device — 'cuda', 'mps' (Mac) или 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def num_workers() -> int:
    """
    Сколько потоков грузить данные.
    На Windows num_workers > 0 требует `if __name__ == "__main__":` в запускающем
    файле (у нас он есть). Если процессор слабый — поставьте 0.
    """
    if os.name == "nt":          # Windows
        return 0                 # безопасно: 0 = грузим в основном потоке
    return min(4, (os.cpu_count() or 2))


def set_seed(seed: int = SEED) -> None:
    """Фиксируем случайность, чтобы результат можно было повторить."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 4. Метаданные модели (classes.json)
# ---------------------------------------------------------------------------
def save_meta(class_names: list[str], extra: dict | None = None) -> None:
    """
    Сохраняем порядок классов и параметры обучения.
    Без этого файла через неделю никто не вспомнит, какой индекс у 'happy'.
    """
    meta = {
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "img_size": IMG_SIZE,
        "channels": NUM_CHANNELS,
        "mean": 0.5,
        "std": 0.5,
    }
    if extra:
        meta.update(extra)
    CLASSES_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta(path: Path | None = None) -> dict:
    """Читаем classes.json. Если файла нет — понятная ошибка с подсказкой."""
    p = Path(path) if path else CLASSES_JSON
    if not p.exists():
        raise FileNotFoundError(
            f"Не найден файл метаданных: {p}\n"
            "Сначала обучите модель: python train.py"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def count_images(folder: Path) -> dict[str, int]:
    """Считает картинки по подпапкам: {'angry': 4953, ...}"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result = {}
    folder = Path(folder)
    if not folder.exists():
        return result
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        result[sub.name] = sum(1 for f in sub.rglob("*") if f.suffix.lower() in exts)
    return result


if __name__ == "__main__":
    print("Папки проекта созданы:")
    for p in (DATA_RAW, DATA, RUNS, CKPT_DIR, ONNX_DIR, MEMES_DIR, REPORTS):
        print("  ", p)
    print("Устройство:", get_device())