from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 128
MIN_MASK_PX = 200

_CH = {"RGB": ("R", "G", "B"), "HSV": ("H", "S", "V")}
_STATS = ("media", "desvio", "p10", "p90")


def _names() -> list[str]:
    n = [f"{esp}_{c}_{st}" for esp, chs in _CH.items() for st in _STATS for c in chs]
    n += ["saturacao_media", "area_mascara", "textura_media", "textura_desvio"]
    for setor in ("ponta", "meio", "base"):
        n += [f"{setor}_saturacao", f"{setor}_brilho"]
    return n


FEATURE_NAMES = _names()


def extract(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if rgb.shape[:2] != (SIZE, SIZE):
        rgb = np.asarray(Image.fromarray(rgb).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
    if mask.shape != (SIZE, SIZE):
        mask = np.asarray(
            Image.fromarray((mask.astype(np.uint8) * 255)).resize((SIZE, SIZE), Image.NEAREST)
        ) > 127
    if mask.sum() < MIN_MASK_PX:
        raise ValueError(f"mascara com {int(mask.sum())} pixels, minimo {MIN_MASK_PX}")

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), np.float32)
    px_rgb = rgb[mask].astype(np.float32)
    px_hsv = hsv[mask]

    f: list[float] = []
    for arr in (px_rgb, px_hsv):
        f += list(arr.mean(0)) + list(arr.std(0))
        f += list(np.percentile(arr, 10, axis=0)) + list(np.percentile(arr, 90, axis=0))

    f += [float(px_hsv[:, 1].mean()), float(mask.mean())]

    grad = np.abs(np.diff(rgb.mean(2), axis=0))
    f += [float(grad.mean()), float(grad.std())]

    for lo, hi in ((0, SIZE // 3), (SIZE // 3, 2 * SIZE // 3), (2 * SIZE // 3, SIZE)):
        faixa = mask.copy()
        faixa[:lo] = False
        faixa[hi:] = False
        if faixa.sum() > 50:
            f += [float(hsv[faixa][:, 1].mean()), float(hsv[faixa][:, 2].mean())]
        else:
            f += [0.0, 0.0]

    v = np.nan_to_num(np.array(f, dtype=np.float64))
    assert len(v) == len(FEATURE_NAMES), f"{len(v)} != {len(FEATURE_NAMES)}"
    return v


def demo() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), bool)
    mask[50:150, 80:220] = True

    v = extract(rgb, mask)
    assert v.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(v).all()

    area = v[FEATURE_NAMES.index("area_mascara")]
    assert 0.18 < area < 0.29, area

    pal = np.full((128, 128, 3), 230, np.uint8)
    sat = np.zeros((128, 128, 3), np.uint8)
    sat[..., 0] = 200
    m = np.ones((128, 128), bool)
    i = FEATURE_NAMES.index("HSV_S_p10")
    assert extract(pal, m)[i] < extract(sat, m)[i]

    try:
        extract(rgb, np.zeros((200, 300), bool))
        raise AssertionError("deveria ter levantado ValueError")
    except ValueError:
        pass

    assert np.allclose(extract(rgb, mask), v)

    print(f"ok - {len(FEATURE_NAMES)} features")


if __name__ == "__main__":
    demo()
