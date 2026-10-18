"""
split_dataset.py — честное разбиение картинок на train / val / test.
 
Универсальный: работает и с готовым датасетом (Работа №1), и с вашим собственным
(Работа №2). Классы берутся из имён папок; папки, начинающиеся с «_» (например
_full), игнорируются; текущие папки train/val/test воспринимаются просто
как «ещё одна папка с картинками класса».
 
Запуск:
    python split_dataset.py --dry-run                     # посмотреть план
    python split_dataset.py                               # FER2013: 80/10/10
    python split_dataset.py --src my_dataset/raw --dst my_dataset --val 0.15 --test 0.15
    python split_dataset.py --src my_dataset/raw --dst my_dataset --min-per-class 20
 
Перемешивание с фиксированным seed=42: повторный запуск даст то же разбиение.
"""
 
from __future__ import annotations
 
import argparse
import random
import shutil
import sys
from pathlib import Path
 
from common import DATA, SEED
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "test", "training", "validation", "valid"}
 
 
def collect(src: Path) -> dict[str, list[Path]]:
    """Собирает картинки по классам, игнорируя служебные папки."""
    buckets: dict[str, list[Path]] = {}
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
            continue
        rel = f.relative_to(src).parts
        if any(part.startswith("_") for part in rel[:-1]):
            continue                      # _full/, _delete_ и прочая служебная мелочь
        cls = f.parent.name
        if cls.lower() in SPLIT_NAMES:    # картинка лежит прямо в train/ и т.п.
            cls = f.parent.parent.name
        if cls.lower() in SPLIT_NAMES:
            continue
        buckets.setdefault(cls.lower(), []).append(f)
    return buckets
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Разбиение датасета на train/val/test")
    parser.add_argument("--src", type=Path, default=DATA,
                        help="где лежат картинки по классам")
    parser.add_argument("--dst", type=Path, default=None,
                        help="куда положить train/val/test (по умолчанию <src>_split)")
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--min-per-class", type=int, default=5,
                        help="минимум картинок в классе, иначе класс пропускаем")
    parser.add_argument("--move", action="store_true", help="перемещать, а не копировать")
    parser.add_argument("--dry-run", action="store_true", help="только показать план")
    args = parser.parse_args()
 
    if args.val + args.test >= 0.95:
        sys.exit("val + test слишком большие — на train ничего не останется")
    src = args.src.expanduser().resolve()
    if not src.exists():
        sys.exit(f"Папки {src} нет.")
    dst = (args.dst or src.parent / (src.name + "_split")).expanduser().resolve()
 
    buckets = collect(src)
    if not buckets:
        sys.exit(f"В {src} не найдено картинок по папкам-классам.")
 
    rng = random.Random(SEED)
    plan: dict[str, dict[str, list[Path]]] = {}
    skipped: list[str] = []
    for cls in sorted(buckets):
        files = buckets[cls][:]
        if len(files) < args.min_per_class:
            skipped.append(f"{cls} ({len(files)} шт.)")
            continue
        rng.shuffle(files)
        n_test = max(1, round(len(files) * args.test)) if args.test > 0 else 0
        n_val = max(1, round(len(files) * args.val)) if args.val > 0 else 0
        plan[cls] = {
            "train": files[:len(files) - n_val - n_test],
            "val": files[len(files) - n_val - n_test:len(files) - n_test],
            "test": files[len(files) - n_test:] if n_test else [],
        }
    if not plan:
        sys.exit("Все классы меньше --min-per-class. Соберите больше данных.")
 
    print("=" * 66)
    print(f"ПЛАН РАЗБИЕНИЯ (seed={SEED})")
    print(f"источник : {src}")
    print(f"куда     : {dst}")
    print("=" * 66)
    print(f"{'класс':<14}{'train':>8}{'val':>8}{'test':>8}{'всего':>8}")
    print("-" * 66)
    totals = {"train": 0, "val": 0, "test": 0}
    for cls in sorted(plan):
        row = {k: len(v) for k, v in plan[cls].items()}
        for k in totals:
            totals[k] += row[k]
        print(f"{cls:<14}{row['train']:>8}{row['val']:>8}{row['test']:>8}"
              f"{sum(row.values()):>8}")
    print("-" * 66)
    print(f"{'ИТОГО':<14}{totals['train']:>8}{totals['val']:>8}{totals['test']:>8}"
          f"{sum(totals.values()):>8}")
    if skipped:
        print("\nПРОПУЩЕНО (мало данных):", ", ".join(skipped))
 
    if args.dry_run:
        print("\n[dry-run] ничего не копируем")
        return
 
    for split in ("train", "val", "test"):
        d = dst / split
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
 
    for cls, parts in plan.items():
        for split, files in parts.items():
            target = dst / split / cls
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                if args.move:
                    shutil.move(str(f), str(target / f.name))
                else:
                    shutil.copy2(f, target / f.name)
 
    print(f"\nГотово: {dst}")
    for split in ("train", "val", "test"):
        d = dst / split
        n = sum(1 for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS)
        print(f"   {split:<6} {n} картинок, {len(list(d.iterdir()))} классов")
    print("\nДальше по методичке.")
 
 
if __name__ == "__main__":
    main()
