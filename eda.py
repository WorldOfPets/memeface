"""
eda.py — разведочный анализ данных (Exploratory Data Analysis).
 
Запуск:
    python eda.py
 
Строит в папке reports/:
    class_balance.png   — гистограмма «сколько картинок на класс»
    samples_grid.png    — 8 случайных примеров каждого класса
    class_mean.png      — «среднее лицо» каждого класса (очень наглядно!)
 
Зачем: прежде чем учить модель, надо понять, ЧЕМУ мы её учим.
"""
 
from __future__ import annotations
 
import random
import sys
from pathlib import Path
 
from common import DATA, REPORTS, SEED, count_images
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def list_images(split: str = "train") -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    root = DATA / split
    if not root.exists():
        root = DATA  # если сплитов нет — берём всё дерево
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in IMG_EXTS and f.is_file():
            cls = f.parent.name
            if cls in {"train", "val", "test"}:
                cls = f.parent.parent.name
            out.setdefault(cls, []).append(f)
    return out
 
 
def plot_balance(by_class: dict[str, list[Path]]) -> None:
    import matplotlib
 
    matplotlib.use("Agg")           # без окна — сохраняем сразу в файл
    import matplotlib.pyplot as plt
 
    classes = sorted(by_class)
    counts = [len(by_class[c]) for c in classes]
 
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(classes, counts, color="#4C8BF5", edgecolor="#1B3A6B")
    ax.set_title("FER2013: сколько картинок на класс", fontsize=13)
    ax.set_ylabel("количество")
    ax.grid(axis="y", alpha=0.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c),
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = REPORTS / "class_balance.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("  сохранено:", out)
 
 
def plot_samples(by_class: dict[str, list[Path]], per_class: int = 8) -> None:
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
 
    classes = sorted(by_class)
    rng = random.Random(SEED)
 
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(per_class * 1.15, len(classes) * 1.25))
    if len(classes) == 1:
        axes = axes[None, :]
 
    for r, cls in enumerate(classes):
        picks = rng.sample(by_class[cls], min(per_class, len(by_class[cls])))
        picks += [None] * (per_class - len(picks))
        for c, path in enumerate(picks):
            ax = axes[r][c]
            if path is not None:
                img = Image.open(path).convert("L")
                ax.imshow(img, cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls, rotation=0, labelpad=42, va="center", fontsize=10)
 
    fig.suptitle("Примеры по классам (8 случайных картинок)", fontsize=13)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    out = REPORTS / "samples_grid.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("  сохранено:", out)
 
 
def plot_class_mean(by_class: dict[str, list[Path]], limit: int = 400) -> None:
    """Среднее изображение класса: видно, куда смотрит модель."""
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
 
    classes = sorted(by_class)
    means = []
    for cls in classes:
        acc = None
        n = 0
        for path in by_class[cls][:limit]:
            img = Image.open(path).convert("L").resize((48, 48))
            data = list(img.getdata())
            if acc is None:
                acc = [0.0] * len(data)
            for i, v in enumerate(data):
                acc[i] += v
            n += 1
        means.append([v / n for v in acc])
 
    fig, axes = plt.subplots(1, len(classes), figsize=(len(classes) * 1.5, 2.0))
    if len(classes) == 1:
        axes = [axes]
    for ax, cls, m in zip(axes, classes, means):
        img = Image.new("L", (48, 48))
        img.putdata([int(v) for v in m])
        ax.imshow(img, cmap="gray")
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    fig.suptitle("«Среднее лицо» каждого класса", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = REPORTS / "class_mean.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("  сохранено:", out)
 
 
def main() -> int:
    if not DATA.exists():
        sys.exit(f"Нет папки {DATA}. Сначала: python download_data.py")
 
    by_class = list_images("train")
    if not by_class:
        sys.exit("Картинки не найдены — проверьте структуру папок")
 
    print("=" * 62)
    print("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
    print("=" * 62)
    total = sum(len(v) for v in by_class.values())
    print(f"Классов: {len(by_class)}, картинок в train: {total}\n")
    for cls in sorted(by_class):
        share = 100 * len(by_class[cls]) / total
        print(f"  {cls:<10} {len(by_class[cls]):>7}   {share:5.1f}%")
 
    print("\nСтроим графики...")
    plot_balance(by_class)
    plot_samples(by_class)
    plot_class_mean(by_class)
 
    print("\nОткройте папку reports/ и посмотрите картинки глазами.")
    print("ВОПРОС СЕБЕ: какой класс модель выучит хуже всего и почему?")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
