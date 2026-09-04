from __future__ import annotations

import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from .data import tabela_mestra, dividir, carregar
from .segmenter import dice_bce, iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(ROOT, "models", "segformer.pt")
BIOHIT = os.path.join(ROOT, "data_ext", "TongeImageDataset")

BACKBONE = "nvidia/mit-b0"
SIZE = 256
EPOCAS = 30
LOTE = 8
LR = 6e-5
PACIENCIA = 8
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


def _norm(rgb: np.ndarray) -> np.ndarray:
    return ((rgb.astype(np.float32) / 255.0 - MEDIA) / DESVIO).transpose(2, 0, 1)


def carregar_biohit(size: int = SIZE):
    X, Y = [], []
    for p in sorted(glob.glob(os.path.join(BIOHIT, "dataset", "*.bmp"))):
        q = os.path.join(BIOHIT, "groundtruth", "mask", os.path.basename(p))
        if not os.path.exists(q):
            continue
        X.append(_norm(np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC))))
        Y.append((np.asarray(Image.open(q).convert("L").resize((size, size), Image.NEAREST)) > 127))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y).astype(np.float32))[:, None]


def carregar_nosso(amostras, size: int = SIZE):
    X = np.stack([_norm(carregar(a, size)[0]) for a in amostras])
    Y = np.stack([carregar(a, size)[1] for a in amostras]).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)[:, None]


def _augment(x, y, rng):
    if rng.random() < 0.5:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    k = int(rng.integers(0, 4))
    if k:
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    if rng.random() < 0.6:
        e = float(rng.uniform(0.6, 1.4))
        n = max(64, int(SIZE * e) // 32 * 32)
        x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(n, n), mode="nearest")
        if n > SIZE:
            o = (n - SIZE) // 2
            x, y = x[:, :, o:o + SIZE, o:o + SIZE], y[:, :, o:o + SIZE, o:o + SIZE]
        else:
            p = SIZE - n
            x = F.pad(x, (0, p, 0, p))
            y = F.pad(y, (0, p, 0, p))
    return x, y


def construir():
    m = SegformerForSemanticSegmentation.from_pretrained(
        BACKBONE, num_labels=1, ignore_mismatched_sizes=True)
    return m


def _logits(modelo, x):
    o = modelo(pixel_values=x).logits
    return F.interpolate(o, size=x.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def avaliar(modelo, X, Y):
    modelo.eval()
    return float(np.mean([iou(torch.sigmoid(_logits(modelo, X[i:i + LOTE])), Y[i:i + LOTE])
                          for i in range(0, len(X), LOTE)]))


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    part = dividir(tabela_mestra())
    Xn, Yn = carregar_nosso(part["treino"])
    Xv, Yv = carregar_nosso(part["val"])
    Xb, Yb = carregar_biohit()
    print("nosso treino=%d  nossa val=%d  biohit=%d" % (len(Xn), len(Xv), len(Xb)), flush=True)

    corte = int(0.9 * len(Xb))
    Xbt, Ybt, Xbv, Ybv = Xb[:corte], Yb[:corte], Xb[corte:], Yb[corte:]
    Xtr = torch.cat([Xn, Xbt])
    Ytr = torch.cat([Yn, Ybt])
    print("treino combinado=%d  (nosso %d + biohit %d)" % (len(Xtr), len(Xn), len(Xbt)), flush=True)

    modelo = construir()
    print("parametros: %s" % f"{sum(p.numel() for p in modelo.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)

    melhor, estado, parado = 0.0, None, 0
    t0 = time.time()
    for ep in range(EPOCAS):
        modelo.train()
        ordem = rng.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(ordem), LOTE):
            idx = ordem[i:i + LOTE]
            xb, yb = _augment(Xtr[idx], Ytr[idx], rng)
            opt.zero_grad()
            perda = dice_bce(_logits(modelo, xb), yb)
            perda.backward()
            opt.step()
            total += perda.item() * len(idx)
        sched.step()

        i_nosso = avaliar(modelo, Xv, Yv)
        i_bio = avaliar(modelo, Xbv, Ybv)
        print("epoca %2d/%d  perda=%.4f  IoU_nosso=%.4f  IoU_biohit=%.4f  (%.0fs)"
              % (ep + 1, EPOCAS, total / len(Xtr), i_nosso, i_bio, time.time() - t0), flush=True)

        if i_nosso > melhor:
            melhor, parado = i_nosso, 0
            estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            melhor_bio = i_bio
        else:
            parado += 1
            if parado >= PACIENCIA:
                print("early stopping", flush=True)
                break

    torch.save({"state_dict": estado, "iou_val": melhor, "iou_biohit": melhor_bio,
                "size": SIZE, "backbone": BACKBONE}, SAIDA)
    print("\nmelhor IoU na NOSSA validacao: %.4f   (U-Net propria: 0.8423)" % melhor, flush=True)
    print("IoU no BioHit reservado:       %.4f   (GA-TongueNet publicado: 0.9814)" % melhor_bio,
          flush=True)
    print("salvo em %s" % SAIDA, flush=True)


if __name__ == "__main__":
    main()
