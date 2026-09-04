from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from .data import tabela_mestra, dividir, carregar
from .segmenter import dice_bce, iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(ROOT, "models", "deeplab.pt")

SIZE = 192
EPOCAS = 40
LOTE = 8
LR = 1e-3
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


def construir() -> torch.nn.Module:
    m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
                           aux_loss=True)
    m.classifier[4] = torch.nn.Conv2d(256, 1, 1)
    m.aux_classifier[4] = torch.nn.Conv2d(256, 1, 1)
    for p in m.backbone.parameters():
        p.requires_grad = False
    return m


def _tensores(amostras):
    X = np.stack([carregar(a, SIZE)[0] for a in amostras]).astype(np.float32) / 255.0
    X = (X - MEDIA) / DESVIO
    Y = np.stack([carregar(a, SIZE)[1] for a in amostras]).astype(np.float32)
    return torch.from_numpy(X.transpose(0, 3, 1, 2)), torch.from_numpy(Y[:, None])


def _augment(x, y, rng):
    if rng.random() < 0.5:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    k = int(rng.integers(0, 4))
    if k:
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    return x, y


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    part = dividir(tabela_mestra())
    Xtr, Ytr = _tensores(part["treino"])
    Xva, Yva = _tensores(part["val"])
    print("treino=%d val=%d" % (len(Xtr), len(Xva)), flush=True)

    modelo = construir()
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    congelados = sum(p.numel() for p in modelo.parameters() if not p.requires_grad)
    print("parametros: %s treinaveis, %s congelados" % (f"{treinaveis:,}", f"{congelados:,}"),
          flush=True)

    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)

    melhor, estado, parado = 0.0, None, 0
    t0 = time.time()
    for ep in range(EPOCAS):
        modelo.train()
        modelo.backbone.eval()
        ordem = rng.permutation(len(Xtr))
        perda_total = 0.0
        for i in range(0, len(ordem), LOTE):
            idx = ordem[i:i + LOTE]
            xb, yb = _augment(Xtr[idx], Ytr[idx], rng)
            opt.zero_grad()
            saida = modelo(xb)
            perda = dice_bce(saida["out"], yb) + 0.4 * dice_bce(saida["aux"], yb)
            perda.backward()
            opt.step()
            perda_total += perda.item() * len(idx)
        sched.step()

        modelo.eval()
        with torch.no_grad():
            ious = [iou(torch.sigmoid(modelo(Xva[i:i + LOTE])["out"]), Yva[i:i + LOTE])
                    for i in range(0, len(Xva), LOTE)]
        iou_val = float(np.mean(ious))
        print("epoca %2d/%d  perda=%.4f  IoU_val=%.4f  (%.0fs)"
              % (ep + 1, EPOCAS, perda_total / len(Xtr), iou_val, time.time() - t0), flush=True)

        if iou_val > melhor:
            melhor, parado = iou_val, 0
            estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            parado += 1
            if parado >= 10:
                print("early stopping", flush=True)
                break

    torch.save({"state_dict": estado, "iou_val": melhor, "size": SIZE}, SAIDA)
    print("\nmelhor IoU de validacao: %.4f" % melhor, flush=True)
    print("U-Net propria, para comparar: 0.8423", flush=True)
    print("salvo em %s" % SAIDA, flush=True)


if __name__ == "__main__":
    main()
