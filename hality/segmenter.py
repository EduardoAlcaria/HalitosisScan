"""Segmentador de lingua: U-Net pequena, treinada nas mascaras do proprio projeto.

Substitui a chamada a API da Roboflow do pipeline anterior, que implicava latencia de
rede por foto, custo por requisicao, chave exposta em texto claro e envio de imagem
clinica a terceiro.

U-Net de ~500k parametros a 192x192. A tarefa e facil -- um blob central grande e de
alto contraste -- entao capacidade nao e o gargalo, e o modelo cabe em CPU.

ponytail: U-Net minima em vez de fine-tune de deeplabv3. 310 pares de treino nao
sustentam um backbone grande, e CPU torna o fine-tune caro. Trocar se o IoU nao bastar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 192


def _block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, w: int = 16):
        super().__init__()
        self.d1, self.d2, self.d3 = _block(3, w), _block(w, w * 2), _block(w * 2, w * 4)
        self.bottom = _block(w * 4, w * 8)
        self.u3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.c3 = _block(w * 8, w * 4)
        self.u2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.c2 = _block(w * 4, w * 2)
        self.u1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.c1 = _block(w * 2, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.d1(x)
        s2 = self.d2(F.max_pool2d(s1, 2))
        s3 = self.d3(F.max_pool2d(s2, 2))
        b = self.bottom(F.max_pool2d(s3, 2))
        y = self.c3(torch.cat([self.u3(b), s3], 1))
        y = self.c2(torch.cat([self.u2(y), s2], 1))
        y = self.c1(torch.cat([self.u1(y), s1], 1))
        return self.out(y)          # logits [N,1,H,W]


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE sozinha ignora o objeto quando o fundo domina; Dice equilibra."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    p, t = pred > 0.5, target > 0.5
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union else 0.0
