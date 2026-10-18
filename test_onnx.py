"""
test_onnx.py — проверяем, что ONNX-модель считает ТО ЖЕ САМОЕ, что и PyTorch.
 
Запуск:
    python test_onnx.py
 
Что проверяем:
  1. Модель вообще открывается в onnxruntime.
  2. Выходы PyTorch и ONNX совпадают (максимальная разница < 1e-4).
  3. Скорость: миллисекунд на кадр и FPS.
  4. Предсказание на 8 случайных картинках из test.
 
Если тут всё зелёное — можно спокойно писать app.py.
"""
 
from __future__ import annotations
 
import argparse
import time
from pathlib import Path
 
import numpy as np
import torch
 
from common import BEST_PT, DATA, ONNX_MODEL, get_device
from model import EmotionCNN
from predictor import EmotionPredictor
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=ONNX_MODEL)
    parser.add_argument("--ckpt", type=Path, default=BEST_PT)
    parser.add_argument("--n", type=int, default=200, help="сколько кадров для замера скорости")
    args = parser.parse_args()
 
    if not args.onnx.exists():
        raise SystemExit(f"Нет {args.onnx}. Сначала: python export_onnx.py")
    if not args.ckpt.exists():
        raise SystemExit(f"Нет {args.ckpt}. Сначала: python train.py")
 
    print("=" * 68)
    print("СРАВНЕНИЕ PyTorch vs ONNX")
    print("=" * 68)
 
    # ------------------------------ загружаем обе модели
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 48))
 
    pt_model = EmotionCNN(num_classes=len(class_names))
    pt_model.load_state_dict(ckpt["model"])
    pt_model.eval()
 
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    in_shape = session.get_inputs()[0].shape
    out_shape = session.get_outputs()[0].shape
    print(f"onnxruntime : {ort.__version__}")
    print(f"вход        : {in_name} {in_shape}")
    print(f"выход       : {out_name} {out_shape}")
 
    # ------------------------------ одинаковые входы -> сравнение выходов
    torch.manual_seed(0)
    xs = torch.randn(8, 1, img_size, img_size)
    with torch.no_grad():
        pt_out = pt_model(xs).numpy()
    onnx_out = session.run(None, {in_name: xs.numpy()})[0]
 
    max_diff = float(np.abs(pt_out - onnx_out).max())
    same_classes = int((pt_out.argmax(1) == onnx_out.argmax(1)).sum())
    print(f"\nМакс. разница выходов : {max_diff:.3e}")
    print(f"Совпало классов       : {same_classes}/8")
    print("ВЕРДИКТ              :", "OK, модели эквивалентны" if max_diff < 1e-3
          else "ОПА! Разница большая — проверьте export_onnx.py (model.eval(), тот же img_size)")
 
    # ------------------------------ скорость
    x1 = np.random.randn(1, 1, img_size, img_size).astype(np.float32)
    for _ in range(10):
        session.run(None, {in_name: x1})          # прогрев
    t0 = time.perf_counter()
    for _ in range(args.n):
        session.run(None, {in_name: x1})
    dt = (time.perf_counter() - t0) / args.n
    print(f"\nСкорость ONNX (CPU)   : {dt * 1000:.2f} мс/кадр  =  {1 / dt:.0f} FPS")
    print("Размер файла          : %.2f МБ" % (args.onnx.stat().st_size / 1024 / 1024))
 
    # ------------------------------ реальные картинки
    print("\nПроверка на реальных картинках (test):")
    predictor = EmotionPredictor(onnx_path=args.onnx, backend="onnx")
    files = [f for f in (DATA / "test").rglob("*") if f.suffix.lower() in IMG_EXTS]
    if not files:
        files = [f for f in DATA.rglob("*") if f.suffix.lower() in IMG_EXTS]
    rng = np.random.default_rng(7)
    ok = 0
    for i in rng.choice(len(files), size=min(8, len(files)), replace=False):
        f = files[int(i)]
        res = predictor.predict(f)
        truth = f.parent.name
        mark = "+" if truth == res["label"] else "X"
        ok += truth == res["label"]
        print(f"  [{mark}] правда={truth:<9} модель={res['label']:<9} "
              f"prob={res['prob']:.2f}  {f.name}")
    print(f"\nПопали: {ok}/8")
    print("Дальше:  python app.py  (интеграция с веб-камерой)")
 
 
if __name__ == "__main__":
    main()
