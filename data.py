"""
data.py — датасет, аугментации и загрузчики (DataLoader).
 
Здесь три главные вещи:
  1. EmotionDataset   — читает картинки из папок <класс>/<файл>
  2. build_transforms — аугментации для train и «честные» преобразования для val/test
  3. make_loaders     — собирает train/val/test-загрузчики и веса классов
 
Аугментация — это «облегчаем себе жизнь»: мы бесплатно делаем датасет в 5-10 раз
разнообразнее (повороты, шум, яркость), и модель перестаёт запоминать конкретные
фотографии.
"""
 
from __future__ import annotations
 
import math
from collections import Counter
from pathlib import Path
 
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T
 
from common import IMG_SIZE
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = 0.5, 0.5   # приводим пиксели из [0..255] в [-1..+1]
 
 
# ---------------------------------------------------------------------------
# Датасет
# ---------------------------------------------------------------------------
class EmotionDataset(Dataset):
    """
    Простой датасет «папки = классы».
 
    Структура:
        root/
          angry/    0001.png ...
          happy/    0001.png ...
    """
 
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Папка датасета не найдена: {self.root}")
 
        # ВАЖНО: sorted() — порядок классов должен быть одинаковым всегда
        self.class_names = sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in p.iterdir())
        )
        if not self.class_names:
            raise RuntimeError(f"В {self.root} нет ни одной подпапки с картинками")
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
 
        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            idx = self.class_to_idx[cls]
            for f in sorted((self.root / cls).rglob("*")):
                if f.suffix.lower() in IMG_EXTS and f.is_file():
                    self.samples.append((f, idx))
 
        self.transform = transform
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, item: int):
        path, label = self.samples[item]
        from PIL import Image, UnidentifiedImageError
 
        try:
            img = Image.open(path).convert("L")     # сразу в grayscale
        except (UnidentifiedImageError, OSError):
            # битый файл: возвращаем соседний, чтобы обучение не падало
            path, label = self.samples[(item + 1) % len(self.samples)]
            img = Image.open(path).convert("L")
 
        if self.transform is not None:
            img = self.transform(img)
        return img, label
 
    def label_counts(self) -> Counter:
        return Counter(label for _, label in self.samples)
 
 
# ---------------------------------------------------------------------------
# Аугментации
# ---------------------------------------------------------------------------
def build_transforms(img_size: int = IMG_SIZE, augment: bool = True):
    """
    Возвращает цепочку преобразований.
 
    augment=True  -> для обучения (крутим, шумим, меняем яркость)
    augment=False -> для val/test (только resize + нормализация: честно!)
    """
    common_tail = [
        T.ToImage(),                              # PIL -> uint8 тензор CHW
        T.ConvertImageDtype(torch.float32),       # -> float [0..1]
        T.Normalize(mean=[MEAN], std=[STD]),      # -> [-1..+1]
    ]
 
    if not augment:
        return T.Compose([T.Resize((img_size, img_size), antialias=True), *common_tail])
 
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        # Лицо не отразить зеркально? Ещё как: люди несимметричны, но для эмоций
        # горизонтальное отражение — самая дешёвая и самая полезная аугментация.
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(12),                                  # наклон головы
        T.RandomAffine(degrees=0, translate=(0.08, 0.08),      # сдвиг
                       scale=(0.9, 1.12)),                     # ближе/дальше
        T.RandomAdjustSharpness(sharpness_factor=1.6, p=0.3),  # резкость
        T.ColorJitter(brightness=0.35, contrast=0.35),         # свет в комнате
        *common_tail,
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),           # «часть лица закрыта»
    ])
 
 
# ---------------------------------------------------------------------------
# Веса классов (боремся с дисбалансом)
# ---------------------------------------------------------------------------
def class_weights(dataset: EmotionDataset) -> torch.Tensor:
    """
    weight_c = N / (K * n_c)
 
    N — всего картинок, K — число классов, n_c — картинок класса c.
    Редкий класс (disgust, 547 шт.) получит большой вес, и модель перестанет
    его игнорировать.
    """
    counts = dataset.label_counts()
    total = sum(counts.values())
    k = len(dataset.class_names)
    weights = [total / (k * counts[i]) for i in range(k)]
    # срезаем экстремумы, чтобы один класс не «перекричал» все остальные
    weights = [min(w, 10.0) for w in weights]
    return torch.tensor(weights, dtype=torch.float32)
 
 
def make_loaders(data_root: str | Path,
                 batch_size: int = 128,
                 img_size: int = IMG_SIZE,
                 num_workers: int = 0,
                 balance: bool = True):
    """
    Собирает загрузчики.
 
    balance=True ->train-выборка формируется WeightedRandomSampler: редкие классы
                   попадаются в батче так же часто, как и популярные.
    Возвращает: train_loader, val_loader, test_loader (или None), weights, class_names
    """
    root = Path(data_root)
    train_tf = build_transforms(img_size, augment=True)
    eval_tf = build_transforms(img_size, augment=False)
 
    train_ds = EmotionDataset(root / "train", train_tf)
    val_dir = root / "val"
    test_dir = root / "test"
 
    val_ds = EmotionDataset(val_dir, eval_tf) if val_dir.exists() else None
    test_ds = EmotionDataset(test_dir, eval_tf) if test_dir.exists() else None
    if test_ds is None and val_ds is not None:
        test_ds = val_ds   # если test не выделили — считаем метрики на val
 
    weights = class_weights(train_ds)
 
    sampler = None
    shuffle = True
    if balance:
        counts = train_ds.label_counts()
        per_sample = [1.0 / counts[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(per_sample, num_samples=len(train_ds), replacement=True)
        shuffle = False    # sampler и shuffle вместе не работают
 
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ) if test_ds else None
 
    return train_loader, val_loader, test_loader, weights, train_ds.class_names
 
 
# ---------------------------------------------------------------------------
# Показать аугментации глазами
# ---------------------------------------------------------------------------
def preview_augmentations(out_path: str | Path = "reports/augmentations.png",
                          n_examples: int = 6) -> None:
    """Сохраняет картинку: слева оригинал, справа N его аугментированных версий."""
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
 
    from common import DATA
 
    tf = build_transforms(IMG_SIZE, augment=True)
    plain = build_transforms(IMG_SIZE, augment=False)
 
    ds = EmotionDataset(DATA / "train", transform=None)
    step = max(1, len(ds) // n_examples)
    picks = [ds.samples[i * step][0] for i in range(n_examples)]
 
    fig, axes = plt.subplots(n_examples, 6, figsize=(9, 1.6 * n_examples))
    for r, path in enumerate(picks):
        img = Image.open(path).convert("L")
        axes[r][0].imshow(_to_np(plain(img)), cmap="gray")
        axes[r][0].set_ylabel(path.parent.name, rotation=0, labelpad=34, va="center")
        for c in range(1, 6):
            axes[r][c].imshow(_to_np(tf(img)), cmap="gray")
        for c in range(6):
            axes[r][c].axis("off")
    fig.suptitle("Аугментации: слева оригинал, справа — что увидит модель", fontsize=12)
    fig.tight_layout(rect=(0.05, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("сохранено:", out_path)
 
 
def _to_np(tensor: torch.Tensor):
    """Тензор (1,H,W) в [-1..1] -> список списков для imshow."""
    t = tensor.squeeze(0)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return [[float(v) for v in row] for row in t.tolist()]
 
 
if __name__ == "__main__":
    # быстрая самопроверка: python data.py
    loaders = make_loaders(Path(__file__).parent / "data" / "fer2013", num_workers=0)
    train_loader, val_loader, test_loader, w, names = loaders
    print("Классы        :", names)
    print("Веса классов  :", [round(float(x), 2) for x in w])
    print("Батчей в train:", len(train_loader))
    x, y = next(iter(train_loader))
    print("Форма батча   :", tuple(x.shape), "метки:", tuple(y.shape))
    print("Диапазон X    :", float(x.min()), "...", float(x.max()))
    print("nan/inf       :", bool(torch.isnan(x).any() or torch.isinf(x).any()))
    preview_augmentations()
    print("math sanity   :", math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-6))
