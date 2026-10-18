"""
train.py — обучение CNN с нуля на FER2013.
 
Запуск:
    python train.py                       # полное обучение (25 эпох)
    python train.py --fast                # быстро: 6 эпох по 20 батчей (для пары)
    python train.py --epochs 40 --lr 3e-4
    python train.py --resume              # продолжить с last.pt
 
Что появляется после обучения:
    checkpoints/best.pt        <- ЛУЧШАЯ модель (её и берём в приложение)
    checkpoints/last.pt        <- модель последней эпохи
    checkpoints/classes.json   <- порядок классов (ОЧЕНЬ ВАЖНО)
    checkpoints/history.json   <- история loss/accuracy по эпохам
    runs/train/curves.png      <- графики обучения
"""
 
from __future__ import annotations
 
import argparse
import json
import time
from itertools import islice
from pathlib import Path
 
import torch
import torch.nn as nn
 
from common import (BATCH_SIZE, BEST_PT, DATA, EPOCHS, HISTORY_JSON, IMG_SIZE,
                    LAST_PT, LR, RUNS, SEED, get_device, num_workers, save_meta,
                    set_seed)
from data import make_loaders
from model import EmotionCNN, count_parameters
 
 
# ---------------------------------------------------------------------------
# Одна эпоха обучения
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device,
                    scheduler=None, log_every=50):
    """
    Классический цикл: forward -> loss -> backward -> step.
    Возвращает (средний loss, accuracy) за эпоху.
    """
    model.train()                      # включаем dropout и обновление статистики BatchNorm
    total_loss, total_ok, total_n = 0.0, 0, 0
 
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
 
        optimizer.zero_grad(set_to_none=True)   # обнуляем старые градиенты
        logits = model(x)                        # прямой проход
        loss = criterion(logits, y)              # функция потерь
        loss.backward()                          # обратный проход (градиенты)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # от «взрыва»
        optimizer.step()                         # шаг оптимизатора
        if scheduler is not None:
            scheduler.step()                     # OneCycleLR шагает на КАЖДЫЙ батч
 
        total_loss += loss.item() * x.size(0)
        total_ok += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
 
        if step % log_every == 0:
            print(f"      batch {step:>4}/{len(loader):<4} "
                  f"loss={loss.item():.4f} acc={total_ok / max(1, total_n):.3f}")
 
    return total_loss / max(1, total_n), total_ok / max(1, total_n)
 
 
# ---------------------------------------------------------------------------
# Проверка на val (без градиентов — быстро)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()                       # dropout выключен, BN использует накопленную статистику
    total_loss, total_ok, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_ok += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return total_loss / max(1, total_n), total_ok / max(1, total_n)
 
 
def plot_curves(history: dict, out_path: Path) -> None:
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["epoch"], history["train_loss"], label="train loss", marker="o", ms=3)
    ax[0].plot(history["epoch"], history["val_loss"], label="val loss", marker="s", ms=3)
    ax[0].set_title("Функция потерь")
    ax[0].set_xlabel("эпоха"); ax[0].legend(); ax[0].grid(alpha=.3)
 
    ax[1].plot(history["epoch"], history["train_acc"], label="train acc", marker="o", ms=3)
    ax[1].plot(history["epoch"], history["val_acc"], label="val acc", marker="s", ms=3)
    ax[1].set_title("Точность")
    ax[1].set_xlabel("эпоха"); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=.3)
 
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("Графики обучения:", out_path)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение CNN эмоций с нуля")
    parser.add_argument("--data", type=Path, default=DATA, help="папка с train/val/test")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--workers", type=int, default=num_workers())
    parser.add_argument("--fast", action="store_true",
                        help="быстрый режим для пары: 6 эпох по 20 батчей")
    parser.add_argument("--resume", action="store_true", help="продолжить с last.pt")
    parser.add_argument("--name", default="train", help="имя папки в runs/")
    args = parser.parse_args()
 
    set_seed(SEED)
    device = get_device()
    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
 
    print("=" * 70)
    print("ОБУЧЕНИЕ CNN С НУЛЯ — MemeFace")
    print("=" * 70)
    print(f"Устройство  : {device}")
    print(f"Данные      : {args.data}")
    print(f"Эпох        : {args.epochs}   batch={args.batch_size}   lr={args.lr}")
    if args.fast:
        args.epochs = 6
        print("РЕЖИМ --fast: 6 эпох по 20 батчей (чтобы успеть на паре)")
 
    # ------------------------------------------------------------------ данные
    train_loader, val_loader, _test_loader, weights, class_names = make_loaders(
        args.data, batch_size=args.batch_size, img_size=args.img_size,
        num_workers=args.workers, balance=True,
    )
    if val_loader is None:
        raise SystemExit("Нет папки val — запустите python split_dataset.py")
 
    steps_per_epoch = len(train_loader)
    if args.fast:
        # берём только первые 20 батчей — purely чтобы увидеть процесс
        train_loader = list(islice(iter(train_loader), 20))
        steps_per_epoch = len(train_loader)
 
    print(f"Классы      : {class_names}")
    print(f"Веса классов: {[round(float(w), 2) for w in weights]}")
    print(f"Батчей      : train={steps_per_epoch}  val={len(val_loader)}")
 
    # ------------------------------------------------------------------ модель
    model = EmotionCNN(num_classes=len(class_names), in_channels=1).to(device)
    print(f"Параметров  : {count_parameters(model):,}  (веса случайные — учим с нуля)")
 
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))   # учитывает дисбаланс
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs),
        pct_start=0.2, div_factor=10.0, final_div_factor=100.0,
    )
 
    start_epoch = 0
    best_acc = 0.0
    history: dict = {"epoch": [], "train_loss": [], "train_acc": [],
                     "val_loss": [], "val_acc": [], "lr": []}
 
    if args.resume and LAST_PT.exists():
        ckpt = torch.load(LAST_PT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", history)
        print(f"Продолжаем с эпохи {start_epoch}, best_acc={best_acc:.4f}")
 
    # ------------------------------------------------------------------ цикл
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        e0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler=scheduler,
            log_every=20 if args.fast else 50,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - e0
 
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
 
        star = ""
        if val_acc > best_acc:
            best_acc = val_acc
            star = "   <- НОВАЯ ЛУЧШАЯ, сохраняем best.pt"
            torch.save({
                "model": model.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size,
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, BEST_PT)
 
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "history": history,
            "class_names": class_names,
            "img_size": args.img_size,
        }, LAST_PT)
 
        print(f"[эпоха {epoch + 1:>3}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={lr_now:.2e} | {dt:.1f} c{star}")
 
    total_time = time.time() - t0
    print("=" * 70)
    print(f"ГОТОВО за {total_time / 60:.1f} мин. Лучшая val accuracy: {best_acc:.4f}")
 
    save_meta(class_names, extra={
        "best_val_acc": best_acc,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "params": count_parameters(model),
        "train_seconds": round(total_time, 1),
        "device": str(device),
        "source": "FER2013 (обучение с нуля)",
    })
    HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_curves(history, run_dir / "curves.png")
 
    print(f"\nЛучшая модель : {BEST_PT}")
    print("Метаданные    : checkpoints/classes.json")
    print("\nДальше:  python evaluate.py")
 
 
if __name__ == "__main__":
    main()
