from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from .segmenter import SIZE as SEG_SIZE, UNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(ROOT, "models")
SAIDA = 128

AREA_PLAUSIVEL_MIN = 0.20
TN_SIZE = 384
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


def realce(rgb: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    o = rgb.astype(np.float32).copy()
    for c in range(3):
        a, b = np.percentile(o[..., c], [lo, hi])
        if b > a:
            o[..., c] = np.clip((o[..., c] - a) * 255 / (b - a), 0, 255)
    return o.astype(np.uint8)


def _redim(m: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (SAIDA, SAIDA), Image.NEAREST)) > 127


class Segmentador:
    def __init__(self) -> None:
        torch.set_num_threads(os.cpu_count() or 4)
        caminho_tn = os.path.join(MODELOS, "tonguenet.pt")
        self.tn = None
        if os.path.exists(caminho_tn):
            from torchvision.models.segmentation import deeplabv3_resnet50
            m = deeplabv3_resnet50(weights=None, weights_backbone=None,
                                   num_classes=2, aux_loss=False)
            sd = torch.load(caminho_tn, map_location="cpu", weights_only=True)
            m.load_state_dict({k.removeprefix("net."): v for k, v in sd.items()})
            self.tn = m.eval()
            self.nome = "tonguenet"
            self.iou_val = 0.9029
        else:
            ck = torch.load(os.path.join(MODELOS, "segmentador.pt"),
                            map_location="cpu", weights_only=True)
            self.unet = UNet(w=ck["w"])
            self.unet.load_state_dict(ck["state_dict"])
            self.unet.eval()
            self.nome = "unet"
            self.iou_val = ck["iou_val"]

    @torch.no_grad()
    def _tongue_net(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (TN_SIZE, TN_SIZE), Image.BILINEAR), np.float32) / 255.0
        x = (x - MEDIA) / DESVIO
        p = self.tn(torch.from_numpy(x.transpose(2, 0, 1))[None])["out"].softmax(1)[0, 1]
        return _redim(p.numpy() > 0.5)

    @torch.no_grad()
    def _unet(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (SEG_SIZE, SEG_SIZE), Image.BICUBIC), np.float32) / 255.0
        p = torch.sigmoid(self.unet(torch.from_numpy(x.transpose(2, 0, 1))[None]))
        return _redim(p[0, 0].numpy() > 0.5)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self.tn is not None:
            return self._tongue_net(rgb)
        m = self._unet(rgb)
        if m.mean() >= AREA_PLAUSIVEL_MIN:
            return m
        r = self._unet(realce(rgb))
        return r if r.mean() > m.mean() else m


def demo() -> None:
    seg = Segmentador()
    rng = np.random.default_rng(0)
    m = seg(rng.integers(0, 256, (400, 300, 3), dtype=np.uint8))
    assert m.shape == (SAIDA, SAIDA) and m.dtype == bool
    esticado = realce(rng.integers(90, 140, (200, 200, 3), dtype=np.uint8))
    assert esticado.max() - esticado.min() > 200
    print("ok - segmentador:", seg.nome)


if __name__ == "__main__":
    demo()
