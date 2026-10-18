"""
check_data.py — проверяем датасет глазами и цифрами ПЕРЕД обучением.
 
Запуск:
    python check_data.py
 
Что делает:
  1. Считает картинки по классам и сплитам.
  2. Ищет битые/пустые файлы (их надо удалить — иначе обучение упадёт на середине).
  3. Проверяет размеры и режим (grayscale/RGB).
  4. Показывает дисбаланс классов.
 
Правило жизни: 80% проблем обучения решается ДО обучения, здесь.
"""
 
from __future__ import annotations
 
import sys
from collections import Counter
from pathlib import Path
 
from common import DATA, count_images
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def human(n: int) -> str:
    return f"{n:,}".replace(",", " ")
 
 
def main() -> int:
    if not DATA.exists():
        sys.exit(f"Папки {DATA} нет. Сначала: python download_data.py")
 
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Нужна Pillow:  uv pip install -U pillow")
 
    print("=" * 66)
    print(f"ПРОВЕРКА ДАТАСЕТА: {DATA}")
    print("=" * 66)
 
    grand_total = 0
    all_classes: set[str] = set()
 
    for split in ("train", "val", "test"):
        split_dir = DATA / split
        if not split_dir.exists():
            print(f"[!] сплита '{split}' нет — пропустим (для обучения хватит train+val)")
            continue
        counts = count_images(split_dir)
        total = sum(counts.values())
        grand_total += total
        all_classes.update(counts)
        print(f"\n--- {split.upper()}  ({human(total)} картинок, {len(counts)} классов) ---")
        for cls, n in counts.items():
            bar = "#" * max(1, round(40 * n / max(counts.values())))
            print(f"  {cls:<10} {human(n):>7}  {bar}")
 
    if grand_total == 0:
        sys.exit("Картинок не найдено. Проверьте, что лежит в data/fer2013")
 
    # ------------------------------------------------------------------
    # Битые файлы и размеры
    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("ТЕХНИЧЕСКАЯ ПРОВЕРКА ФАЙЛОВ (может занять пару минут)")
    print("=" * 66)
    bad: list[Path] = []
    sizes: Counter = Counter()
    modes: Counter = Counter()
    checked = 0
 
    for f in DATA.rglob("*"):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        checked += 1
        try:
            with Image.open(f) as im:
                im.verify()                     # проверка целостности
            with Image.open(f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception:                       # noqa: BLE001
            bad.append(f)
        if checked % 5000 == 0:
            print(f"  проверено {human(checked)} ...")
 
    print(f"\nПроверено файлов : {human(checked)}")
    print(f"Битых файлов     : {len(bad)}")
    if bad:
        for f in bad[:15]:
            print("   [X]", f)
        print("Удаляем их командой:")
        print("   python -c \"from pathlib import Path; [f.unlink() for f in Path('data/fer2013').rglob('*') if f.suffix.lower() in {'.jpg','.png','.jpeg'} and f.stat().st_size < 100]\"")
        print("(или удалите вручную — список выше)")
 
    print("\nРазмеры изображений (топ-5):")
    for size, n in sizes.most_common(5):
        print(f"   {size[0]}x{size[1]} : {human(n)}")
    print("Режимы цвета:")
    for mode, n in modes.most_common():
        meaning = {"L": "grayscale (ч/б)", "RGB": "цветное", "RGBA": "цветное + прозрачность"}.get(mode, mode)
        print(f"   {mode:<5} : {human(n)}  ({meaning})")
 
    # ------------------------------------------------------------------
    # Дисбаланс
    # ------------------------------------------------------------------
    train_counts = count_images(DATA / "train")
    if train_counts:
        mx = max(train_counts.values())
        mn = min(train_counts.values())
        print("\n" + "=" * 66)
        print("ДИСБАЛАНС КЛАССОВ (train)")
        print("=" * 66)
        print(f"Самый большой класс  : {max(train_counts, key=train_counts.get)} ({human(mx)})")
        print(f"Самый маленький класс: {min(train_counts, key=train_counts.get)} ({human(mn)})")
        print(f"Разница              : в {mx / max(1, mn):.1f} раз")
        print("\nВывод: без взвешенной функции потерь модель будет угадывать 'happy'")
        print("и почти никогда не предсказывать 'disgust'. Мы это учтём в train.py")
 
    print("\nВсё посчитано. Дальше:  python eda.py   (посмотреть глазами)")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
