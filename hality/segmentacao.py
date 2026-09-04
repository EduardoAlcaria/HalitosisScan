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
AREA_SAM_MIN = 0.15


def realce(rgb: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    o = rgb.astype(np.float32).copy()
    for c in range(3):
        a, b = np.percentile(o[..., c], [lo, hi])
        if b > a:
            o[..., c] = np.clip((o[..., c] - a) * 255 / (b - a), 0, 255)
    return o.astype(np.uint8)


class Segmentador:
    def __init__(self, usar_sam: bool = True) -> None:
        ck = torch.load(os.path.join(MODELOS, "segmentador.pt"),
                        map_location="cpu", weights_only=True)
        self.unet = UNet(w=ck["w"])
        self.unet.load_state_dict(ck["state_dict"])
        self.unet.eval()
        self.iou_val = ck["iou_val"]
        self._sam = None
        self.usar_sam = usar_sam and os.path.exists(
            os.path.join(MODELOS, "sam", "sam_encoder_vit_b.onnx"))
        self.n_sam = 0

    @torch.no_grad()
    def _unet(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (SEG_SIZE, SEG_SIZE), Image.BICUBIC), np.float32) / 255.0
        p = torch.sigmoid(self.unet(torch.from_numpy(x.transpose(2, 0, 1))[None]))
        m = (p[0, 0].numpy() > 0.5).astype(np.uint8) * 255
        return np.asarray(Image.fromarray(m).resize((SAIDA, SAIDA), Image.NEAREST)) > 127

    def _sam_modelo(self):
        if self._sam is None:
            from .sam_onnx import SamGPU
            self._sam = SamGPU()
        return self._sam

    def _via_sam(self, rgb: np.ndarray, semente: np.ndarray) -> np.ndarray | None:
        try:
            H, W = rgb.shape[:2]
            sem = np.asarray(Image.fromarray((semente.astype(np.uint8) * 255)).resize(
                (W, H), Image.NEAREST)) > 127
            m = self._sam_modelo().isolar(rgb, sem)
        except Exception:
            return None
        if m is None:
            return None
        self.n_sam += 1
        return np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize(
            (SAIDA, SAIDA), Image.NEAREST)) > 127

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        m = self._unet(rgb)
        if m.mean() >= AREA_PLAUSIVEL_MIN:
            return m

        r = self._unet(realce(rgb))
        if r.mean() > m.mean():
            m = r
        if m.mean() >= AREA_PLAUSIVEL_MIN or not self.usar_sam:
            return m

        if m.mean() < AREA_SAM_MIN:
            s = self._via_sam(rgb, m)
            if s is not None and s.mean() > m.mean():
                return s
        return m


def demo() -> None:
    seg = Segmentador(usar_sam=False)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)
    m = seg(rgb)
    assert m.shape == (SAIDA, SAIDA) and m.dtype == bool

    plano = np.full((200, 200, 3), 120, np.uint8)
    assert realce(plano).shape == plano.shape
    esticado = realce(rng.integers(90, 140, (200, 200, 3), dtype=np.uint8))
    assert esticado.max() - esticado.min() > 200

    print("ok - cascata de segmentacao")


if __name__ == "__main__":
    demo()
