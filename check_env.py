"""
check_env.py — проверка окружения перед стартом.
 
Запуск:  python check_env.py
 
Скрипт проходит по всем библиотекам проекта и честно говорит, что установлено,
а что нет. Если что-то не так — читаем подсказку и чиним, а не гадаем.
"""
 
import platform
import sys
from pathlib import Path
 
OK = "[OK]"
NO = "[--]"
 
 
def check_import(name: str, show_version: bool = True) -> bool:
    """Пробует импортировать библиотеку и печатает её версию."""
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "?") if show_version else ""
        print(f"{OK} {name:<16} {version}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{NO} {name:<16} НЕ УСТАНОВЛЕН ({type(exc).__name__})")
        return False
 
 
def main() -> int:
    print("=" * 62)
    print("ПРОВЕРКА ОКРУЖЕНИЯ MemeFace")
    print("=" * 62)
    print(f"Python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"ОС       : {platform.system()} {platform.release()}")
    print(f"Процессор: {platform.processor() or 'не определён'}")
    print("-" * 62)
 
    results = {}
    for lib in ("torch", "torchvision", "cv2", "numpy", "PIL", "matplotlib", "tqdm"):
        results[lib] = check_import(lib)
    print("-" * 62)
 
    # GPU / CPU
    try:
        import torch
 
        print(f"torch собран с CUDA : {torch.version.cuda}")
        print(f"CUDA доступна       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Видеокарта          : {torch.cuda.get_device_name(0)}")
        else:
            print("Видеокарта не найдена -> обучаемся на CPU (это нормально, просто дольше)")
    except Exception as exc:  # noqa: BLE001
        print("Не удалось проверить torch:", exc)
 
    # Веб-камера
    try:
        import cv2
 
        cap = cv2.VideoCapture(0)
        got, frame = cap.read()
        cap.release()
        if got:
            print(f"{OK} Веб-камера        : работает, кадр {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"{NO} Веб-камера        : не открылась (проверьте доступ/другое приложение)")
    except Exception as exc:  # noqa: BLE001
        print(f"{NO} Веб-камера        : ошибка {exc}")
 
    # Данные
    data_dir = Path(__file__).resolve().parent / "data" / "fer2013"
    if data_dir.exists():
        n = sum(1 for _ in data_dir.rglob("*") if _.suffix.lower() in {".jpg", ".png", ".jpeg"})
        print(f"{OK} Датасет           : {data_dir} ({n} картинок)")
    else:
        print(f"{NO} Датасет           : {data_dir} не найден -> python download_data.py")
 
    print("-" * 62)
    missing = [k for k, v in results.items() if not v]
    if missing:
        print("НЕ ХВАТАЕТ:", ", ".join(missing))
        print("Установите их командой из Задачи 1 и запустите проверку снова.")
        return 1
    print("Всё на месте. Можно идти дальше -> python download_data.py")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
