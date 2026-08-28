"""Treina a U-Net de segmentacao nas mascaras do proprio projeto.

Treina SO na particao de treino. Usar imagens de teste aqui vazaria: o segmentador
produziria mascaras melhores justamente nas fotos onde o classificador sera medido.

As mascaras de referencia vieram da API da Roboflow. Isto e destilacao: herdamos o teto
de qualidade dela, mas eliminamos a dependencia de rede, o custo por chamada e o envio
de imagem clinica a terceiro.

Augmentation: apenas geometrica. Medicao em docs/ARQUITETURA.md secao 5.2 -- jitter de
cor degrada monotonicamente, porque a cor e o sinal.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from PIL import Image

from .data import tabela_mestra, dividir, carregar
from .segmenter import SIZE, UNet, dice_bce, iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(ROOT, "models", "segmentador.pt")

EPOCAS = 60
LOTE = 8
LR = 3e-3


def _tensores(amostras, size=SIZE):
    X = np.stack([carregar(a, size)[0] for a in amostras]).astype(np.float32) / 255.0
    Y = np.stack([carregar(a, size)[1] for a in amostras]).astype(np.float32)
    return torch.from_numpy(X.transpose(0, 3, 1, 2)), torch.from_numpy(Y[:, None])


def _augment(x: torch.Tensor, y: torch.Tensor, rng) -> tuple[torch.Tensor, torch.Tensor]:
    if rng.random() < 0.5:                       # espelhamento horizontal
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    k = int(rng.integers(0, 4))
    if k:                                        # rotacao em multiplos de 90
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    return x, y


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    part = dividir(tabela_mestra())
    print(f"treino={len(part['treino'])}  val={len(part['val'])}", flush=True)

    Xtr, Ytr = _tensores(part["treino"])
    Xva, Yva = _tensores(part["val"])
    print(f"tensores: {tuple(Xtr.shape)}", flush=True)

    modelo = UNet(w=16)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"parametros: {n_par:,}", flush=True)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)

    melhor, melhor_estado, sem_melhora = 0.0, None, 0
    t0 = time.time()
    for ep in range(EPOCAS):
        modelo.train()
        ordem = rng.permutation(len(Xtr))
        perda_total = 0.0
        for i in range(0, len(ordem), LOTE):
            idx = ordem[i:i + LOTE]
            xb, yb = _augment(Xtr[idx], Ytr[idx], rng)
            opt.zero_grad()
            perda = dice_bce(modelo(xb), yb)
            perda.backward()
            opt.step()
            perda_total += perda.item() * len(idx)
        sched.step()

        modelo.eval()
        with torch.no_grad():
            ious = [iou(torch.sigmoid(modelo(Xva[i:i + LOTE])), Yva[i:i + LOTE])
                    for i in range(0, len(Xva), LOTE)]
        iou_val = float(np.mean(ious))
        print(f"epoca {ep + 1:3d}/{EPOCAS}  perda={perda_total / len(Xtr):.4f}  "
              f"IoU_val={iou_val:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        if iou_val > melhor:
            melhor, sem_melhora = iou_val, 0
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= 12:
                print("early stopping", flush=True)
                break

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    torch.save({"state_dict": melhor_estado, "iou_val": melhor, "size": SIZE, "w": 16}, SAIDA)
    print(f"\nmelhor IoU de validacao: {melhor:.4f}", flush=True)
    print(f"salvo em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
