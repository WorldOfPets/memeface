"""
download_data.py — скачиваем ГОТОВЫЙ РАЗМЕЧЕННЫЙ датасет FER2013.
 
FER2013: 35 887 чёрно-белых фото лиц 48x48, 7 эмоций.
Разметка уже есть — это то, что нам нужно для Работы №1.
 
Запуск:
    python download_data.py                # способ А (Hugging Face, без регистрации)
    python download_data.py --source kaggle  # способ Б (нужен аккаунт Kaggle)
 
Способ А (рекомендуем): репозиторий manojdilz/facial_emotion_detection_dataset —
это FER2013, уже разложенный по папкам train/<эмоция> и test/<эмоция>.
 
Способ Б: официальный fer2013.csv с Kaggle (msambare/facial-emotion-recognition-2013).
Тогда после скачивания запустите:  python convert_csv.py
 
Если интернет закрыт — скачайте архив дома и распакуйте в data/raw вручную,
дальше скрипт сам всё разложит по папкам.
"""
 
from __future__ import annotations
 
import argparse
import shutil
import sys
from pathlib import Path
 
from common import DATA, DATA_RAW
 
HF_REPO = "manojdilz/facial_emotion_detection_dataset"
KAGGLE_DATASET = "msambare/facial-emotion-recognition-2013"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
# ---------------------------------------------------------------------------
# Способ А: Hugging Face
# ---------------------------------------------------------------------------
def download_from_hf() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "Не установлена библиотека huggingface_hub.\n"
            "Выполните:  uv pip install -U huggingface_hub"
        )
 
    target = DATA_RAW / "hf_fer2013"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваем {HF_REPO} ...")
    print("(около 300-500 МБ, скорость зависит от интернета — идём пить чай)")
    path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(target),
        # качаем только картинки и ничего лишнего
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "train/**", "test/**"],
    )
    print("Готово:", path)
    return Path(path)
 
 
# ---------------------------------------------------------------------------
# Способ Б: Kaggle
# ---------------------------------------------------------------------------
def download_from_kaggle() -> Path:
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "Не установлена библиотека kagglehub.\n"
            "Выполните:  uv pip install -U kagglehub\n"
            "И не забудьте положить kaggle.json в ~/.kaggle/ (инструкция в методичке)."
        )
    print(f"Скачиваем {KAGGLE_DATASET} ...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Готово:", path)
    return Path(path)
 
 
# ---------------------------------------------------------------------------
# Приводим всё к одной структуре: data/fer2013/{train,val,test}/<класс>/*.jpg
# ---------------------------------------------------------------------------
def find_class_folders(root: Path) -> dict[str, list[Path]]:
    """
    Ищет в скачанном дереве папки, похожие на классы эмоций.
    Возвращает {'train': [Path('angry'), ...], 'val': [...], 'test': [...]}
    """
    splits: dict[str, list[Path]] = {}
    for split_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        name = split_dir.name.lower()
        if name not in {"train", "training", "val", "valid", "validation", "test",
                        "publictest", "privatetest", "private"}:
            continue
        # внутри должны быть папки-классы
        class_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
        if not class_dirs:
            continue
        key = "train" if name in {"train", "training"} else (
            "val" if name in {"val", "valid", "validation", "publictest"} else "test")
        splits.setdefault(key, []).extend(class_dirs)
    return splits
 
 
def normalize_names(name: str) -> str:
    """happy/Happy/HAPPY -> happy; privateTest -> test handled elsewhere."""
    n = name.strip().lower()
    aliases = {
        "happiness": "happy", "sadness": "sad", "anger": "angry",
        "fearful": "fear", "surprised": "surprise", "disgusted": "disgust",
    }
    return aliases.get(n, n)
 
 
def build_dataset(src: Path) -> None:
    splits = find_class_folders(src)
    if not splits:
        sys.exit(
            f"В {src} не найдена структура <split>/<класс>/картинки.\n"
            "Посмотрите глазами, что скачалось, и поправьте пути в этом скрипте\n"
            "(или распакуйте архив вручную в data/raw)."
        )
 
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
 
    total = 0
    for split, class_dirs in splits.items():
        for cdir in class_dirs:
            cls = normalize_names(cdir.name)
            dst = DATA / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            moved = 0
            for f in sorted(cdir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    # копируем (не перемещаем), чтобы исходник остался целым
                    target = dst / f.name
                    if not target.exists():
                        shutil.copy2(f, target)
                    moved += 1
            total += moved
            print(f"  {split:<6} {cls:<10} -> {moved} файлов")
 
    print("-" * 50)
    print(f"Всего картинок: {total}")
    print(f"Датасет лежит в: {DATA}")
    print("Дальше:  python check_data.py")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать датасет FER2013")
    parser.add_argument("--source", choices=["hf", "kaggle"], default="hf",
                        help="hf = Hugging Face (по умолчанию), kaggle = Kaggle")
    parser.add_argument("--local", type=Path, default=None,
                        help="не качать, а взять уже распакованную папку")
    args = parser.parse_args()
 
    if args.local:
        src = Path(args.local).expanduser().resolve()
        if not src.exists():
            sys.exit(f"Папка не найдена: {src}")
    elif args.source == "kaggle":
        src = download_from_kaggle()
    else:
        src = download_from_hf()
 
    build_dataset(Path(src))
 
 
if __name__ == "__main__":
    main()
