"""
export_onnx.py — превращаем PyTorch-модель (best.pt) в ONNX (emotion.onnx).
 
Зачем: ONNX — это «универсальный контейнер» для нейросетей. Такую модель умеют
запускать onnxruntime, OpenCV DNN, X-AnyLabeling, мобильные приложения — без
установки тяжёлого PyTorch.
 
Запуск:
    python export_onnx.py
    python export_onnx.py --opset 17 --dynamic
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import torch
 
from common import BEST_PT, CKPT_DIR, ONNX_DIR, ONNX_MODEL, save_meta
from model import EmotionCNN
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=BEST_PT)
    parser.add_argument("--out", type=Path, default=ONNX_MODEL)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", default=True,
                        help="разрешить любой размер батча (по умолчанию включено)")
    parser.add_argument("--static", dest="dynamic", action="store_false",
                        help="фиксированный батч = 1 (нужен для некоторых конвертеров)")
    args = parser.parse_args()
 
    if not args.ckpt.exists():
        raise SystemExit(f"Нет чекпоинта {args.ckpt}. Сначала: python train.py")
 
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 48))
 
    model = EmotionCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model"])
    model.eval()                                    # ВАЖНО: dropout/BN в режим инференса
 
    dummy = torch.randn(1, 1, img_size, img_size)
 
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
 
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Экспортирую {args.ckpt.name} -> {args.out.name} (opset {args.opset})")
 
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        # если ваш torch >= 2.9 и экспорт ругается — раскомментируйте строку ниже:
        # dynamo=False,
    )
 
    # Сохраняем метаданные рядом с моделью — без них никто не вспомнит порядок классов
    save_meta(class_names, extra={
        "val_acc": ckpt.get("val_acc"),
        "source_ckpt": str(args.ckpt),
        "opset": args.opset,
        "input_name": "input",
        "output_name": "logits",
    })
 
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"Готово: {args.out}  ({size_mb:.2f} МБ)")
    print(f"Метаданные: {CKPT_DIR / 'classes.json'}")
    print("\nПроверьте модель глазами на https://netron.app (перетащите файл в браузер)")
    print("Дальше:  python test_onnx.py")
 
 
if __name__ == "__main__":
    main()
