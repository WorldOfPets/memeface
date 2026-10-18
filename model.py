"""
model.py — свёрточная нейросеть (CNN) для 7 эмоций. ОБУЧАЕМ С НУЛЯ.
 
Вход : тензор (B, 1, 48, 48)  — чёрно-белое лицо 48x48
Выход: тензор (B, 7)            — «сырые» очки (logits) по каждому классу
 
Архитектура (классика, которая честно работает на FER2013):
 
    Conv(1->32)  BN ReLU
    Conv(32->32) BN ReLU   MaxPool 2x2  Drop   48x48 -> 24x24
    Conv(32->64) BN ReLU
    Conv(64->64) BN ReLU   MaxPool 2x2  Drop   24x24 -> 12x12
    Conv(64->128) BN ReLU
    Conv(128->128) BN ReLU MaxPool 2x2  Drop   12x12 -> 6x6
    Flatten (128*6*6 = 4608)
    Linear 4608->256  BN ReLU Drop
    Linear 256->7
 
Параметров ~1.4 млн. На CPU это обучается за 20-40 минут, на GPU — за 5-10.
"""
 
from __future__ import annotations
 
import torch
import torch.nn as nn
 
 
def conv_block(in_ch: int, out_ch: int, pool: bool = True, drop: float = 0.25) -> nn.Sequential:
    """
    «Кирпичик» из двух свёрток 3x3 + BatchNorm + ReLU, затем пулинг и dropout.
 
    padding=1 сохраняет размер (48 -> 48), пулинг уменьшает вдвое (48 -> 24).
    Чем глубже слой, тем более крупные черты он видит:
      блок 1 — edges/углы, блок 2 — глаза/нос/рот, блок 3 — выражение целиком.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers += [nn.MaxPool2d(2), nn.Dropout2d(drop)]
    return nn.Sequential(*layers)
 
 
class EmotionCNN(nn.Module):
    """Классификатор эмоций. Обучается с нуля (веса случайные)."""
 
    def __init__(self, num_classes: int = 7, in_channels: int = 1, base: int = 32):
        super().__init__()
        self.num_classes = num_classes
 
        self.features = nn.Sequential(
            conv_block(in_channels, base, pool=True),        # 1 -> 32,  48 -> 24
            conv_block(base, base * 2, pool=True),           # 32 -> 64, 24 -> 12
            conv_block(base * 2, base * 4, pool=True),       # 64 -> 128,12 -> 6
        )
        flat = base * 4 * 6 * 6                              # 128 * 36 = 4608
 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                                 # dropout — защита от переобучения
            nn.Linear(256, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x            # logits: softmax применяется внутри CrossEntropyLoss
 
 
def count_parameters(model: nn.Module) -> int:
    """Сколько обучаемых параметров (весов) в модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
 
def describe_shapes(model: nn.Module, img_size: int = 48) -> None:
    """Печатает форму тензора после каждого блока — удобно для понимания."""
    model.eval()                       # BN не считает статистику на батче из 1
    with torch.no_grad():
        x = torch.zeros(1, 1, img_size, img_size)
        print(f"вход           : {tuple(x.shape)}")
        for i, block in enumerate(model.features, start=1):
            x = block(x)
            print(f"блок {i}         : {tuple(x.shape)}")
        x = model.classifier(x)
        print(f"выход (logits) : {tuple(x.shape)}")
    model.train()
 
 
if __name__ == "__main__":
    model = EmotionCNN(num_classes=7)
    print(model)
    print("-" * 60)
    print("Обучаемых параметров:", f"{count_parameters(model):,}")
    describe_shapes(model)
 
    # мини-проверка: прямой проход и обратный
    out = model(torch.randn(4, 1, 48, 48))
    loss = nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    print("loss =", round(float(loss), 4), "| градиенты посчитаны:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))
