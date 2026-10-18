"""
evaluate.py — честная оценка обученной модели.
 
Запуск:
    python evaluate.py                 # оценка на test (или val, если test нет)
    python evaluate.py --split val
    python evaluate.py --show-errors 12
 
Строит в reports/:
    confusion_matrix.png  — матрица ошибок (ГЛАВНАЯ картинка работы)
    per_class_report.txt  — precision / recall / F1 по классам
    errors_grid.png       — примеры, где модель ошиблась
"""
 
from __future__ import annotations
 
import argparse
from collections import defaultdict
from pathlib import Path
 
import numpy as np
import torch
 
from common import BEST_PT, DATA, REPORTS, get_device
from data import EmotionDataset, build_transforms
from model import EmotionCNN
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
# ---------------------------------------------------------------------------
# Метрики без sklearn (чтобы не ставить лишнее)
# ---------------------------------------------------------------------------
def prf1(y_true: list[int], y_pred: list[int], num_classes: int):
    """Возвращает списки precision, recall, f1 и support для каждого класса."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
 
    precision, recall, f1, support = [], [], [], []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precision.append(prec); recall.append(rec); f1.append(f)
        support.append(tp[c] + fn[c])
    return precision, recall, f1, support
 
 
def plot_confusion(cm: np.ndarray, class_names: list[str], out: Path) -> None:
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=40, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("предсказание модели")
    ax.set_ylabel("правильный ответ")
    ax.set_title("Матрица ошибок (нормирована по строкам)")
 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = norm[i, j]
            ax.text(j, i, f"{val:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.55 else "#222222")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
 
 
def plot_errors(items: list[tuple], class_names: list[str], out: Path) -> None:
    """items = [(path, true_idx, pred_idx, prob), ...]"""
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
 
    n = len(items)
    if n == 0:
        print("Ошибок нет (подозрительно!)")
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes = np.array(axes).reshape(-1)
 
    for ax in axes:
        ax.axis("off")
    for ax, (path, t, p, prob) in zip(axes, items):
        img = Image.open(path).convert("L")
        ax.imshow(img, cmap="gray")
        ax.set_title(f"правда: {class_names[t]}\nмодель: {class_names[p]} ({prob:.2f})",
                     fontsize=7, color="#B00020")
        ax.axis("off")
    fig.suptitle("Где модель ошиблась", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)
 
 
@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--ckpt", type=Path, default=BEST_PT)
    parser.add_argument("--show-errors", type=int, default=12)
    args = parser.parse_args()
 
    device = get_device()
    split_dir = args.data / args.split
    if not split_dir.exists():
        split_dir = args.data / "val"
        print(f"[!] {args.split} не найден, беру val")
    if not split_dir.exists():
        raise SystemExit("Нет ни test, ни val — запустите split_dataset.py")
 
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 48))
 
    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
 
    ds = EmotionDataset(split_dir, transform=build_transforms(img_size, augment=False))
    print("=" * 68)
    print(f"ОЦЕНКА МОДЕЛИ на {split_dir.name}: {len(ds)} картинок, "
          f"{len(class_names)} классов")
    print("=" * 68)
 
    y_true: list[int] = []
    y_pred: list[int] = []
    probs_pred: list[float] = []
    errors: list[tuple] = []
    per_class_hits: dict[int, list[int]] = defaultdict(list)
 
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    index = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        pred = pred.cpu()
        for b in range(x.size(0)):
            t = int(y[b]); p = int(pred[b]); c = float(conf[b])
            y_true.append(t); y_pred.append(p); probs_pred.append(c)
            per_class_hits[t].append(int(t == p))
            if t != p and len(errors) < 4000:
                errors.append((ds.samples[index][0], t, p, c))
            index += 1
 
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    precision, recall, f1, support = prf1(y_true, y_pred, len(class_names))
    macro_f1 = float(np.mean(f1))
 
    print(f"\nAccuracy          : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Средняя уверенность правильных ответов: "
          f"{np.mean([p for p, a, b in zip(probs_pred, y_true, y_pred) if a == b]):.3f}")
    print(f"Средняя уверенность ошибок          : "
          f"{np.mean([p for p, a, b in zip(probs_pred, y_true, y_pred) if a != b] or [0]):.3f}")
 
    print(f"\n{'класс':<12}{'precision':>10}{'recall':>9}{'F1':>8}{'support':>9}")
    print("-" * 48)
    lines = [f"{'класс':<12}{'precision':>10}{'recall':>9}{'F1':>8}{'support':>9}",
             "-" * 48]
    for i, cls in enumerate(class_names):
        row = (f"{cls:<12}{precision[i]:>10.3f}{recall[i]:>9.3f}{f1[i]:>8.3f}{support[i]:>9}")
        print(row)
        lines.append(row)
    lines += ["-" * 48, f"accuracy = {acc:.4f}", f"macro-F1 = {macro_f1:.4f}"]
    (REPORTS / "per_class_report.txt").write_text("\n".join(lines), encoding="utf-8")
 
    # матрица ошибок
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plot_confusion(cm, class_names, REPORTS / "confusion_matrix.png")
    print("\nМатрица ошибок: reports/confusion_matrix.png")
 
    # примеры ошибок
    if args.show_errors:
        rng = np.random.default_rng(42)
        picks = [errors[i] for i in rng.choice(len(errors),
                                                size=min(args.show_errors, len(errors)),
                                                replace=False)] if errors else []
        plot_errors(picks, class_names, REPORTS / "errors_grid.png")
        print("Примеры ошибок: reports/errors_grid.png")
 
    # главный вывод
    print("\n" + "=" * 68)
    print("КАК ЧИТАТЬ РЕЗУЛЬТАТ")
    print("=" * 68)
    print(" * 0.65-0.73 accuracy на FER2013 — это НОРМАЛЬНО (человек ~0.65, SOTA ~0.76).")
    print(" * Класс 'disgust' почти всегда провален: в датасете всего ~547 картинок.")
    print(" * Смотрите на ДИАГОНАЛЬ матрицы ошибок — чем она темнее, тем лучше.")
    print(" * Если train_acc >> val_acc — это переобучение: добавьте аугментаций/dropout.")
    print("\nДальше:  python export_onnx.py")
 
 
if __name__ == "__main__":
    main()
