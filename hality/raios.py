from __future__ import annotations

import numpy as np
from PIL import Image

N_RAIOS = 180
PASSO = 1.5
CONFIRMA = 4
RAIO_MIN_FRAC = 0.03
RAIO_MAX_FRAC = 0.75
K_SUAVIZA = 9


def _lab(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    m = x > 0.04045
    x = np.where(m, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(np.maximum(t, 1e-9)), 7.787 * t + 16 / 116)

    fx, fy, fz = f(X), f(Y), f(Z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def _estat(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(p, 0)
    mad = np.maximum(np.median(np.abs(p - med), 0) * 1.4826, [2.0, 1.0, 1.0])
    return med, 1.0 / mad


def modelos(lab: np.ndarray, semente: np.ndarray):
    dentro = lab[semente]
    fora_m = _borda(semente)
    fora = lab[fora_m] if fora_m.sum() > 50 else lab[~semente]
    return _estat(dentro), _estat(fora)


def _borda(semente: np.ndarray, frac: float = 0.10) -> np.ndarray:
    H, W = semente.shape
    b = np.zeros((H, W), bool)
    k = max(2, int(min(H, W) * frac))
    b[:k] = b[-k:] = True
    b[:, :k] = b[:, -k:] = True
    return b & ~semente


def _amostra(lab: np.ndarray, y: float, x: float) -> np.ndarray | None:
    H, W = lab.shape[:2]
    yi, xi = int(round(y)), int(round(x))
    if 0 <= yi < H and 0 <= xi < W:
        return lab[yi, xi]
    return None


def contorno_por_raios(rgb: np.ndarray, semente_mask: np.ndarray,
                       semente: tuple[int, int]) -> np.ndarray:
    lab = _lab(rgb)
    H, W = rgb.shape[:2]
    cy, cx = semente
    (m_l, i_l), (m_f, i_f) = modelos(lab, semente_mask)

    r_min = RAIO_MIN_FRAC * min(H, W)
    r_max = RAIO_MAX_FRAC * min(H, W)
    angs = np.linspace(0, 2 * np.pi, N_RAIOS, endpoint=False)
    raios = np.full(N_RAIOS, r_max, np.float32)

    for k, a in enumerate(angs):
        dy, dx = np.sin(a), np.cos(a)
        fora = 0
        r = r_min
        while r < r_max:
            v = _amostra(lab, cy + dy * r, cx + dx * r)
            if v is None:
                raios[k] = r
                break
            d_lingua = np.sqrt((((v - m_l) * i_l) ** 2).sum())
            d_fora = np.sqrt((((v - m_f) * i_f) ** 2).sum())
            if d_fora < d_lingua:
                fora += 1
                if fora >= CONFIRMA:
                    raios[k] = r - CONFIRMA * PASSO
                    break
            else:
                fora = 0
            r += PASSO
        else:
            raios[k] = r_max

    pad = np.r_[raios[-K_SUAVIZA:], raios, raios[:K_SUAVIZA]]
    suave = np.array([np.median(pad[i:i + 2 * K_SUAVIZA + 1]) for i in range(N_RAIOS)])
    suave = np.clip(suave, r_min, r_max)

    ys = cy + np.sin(angs) * suave
    xs = cx + np.cos(angs) * suave
    return _preencher(np.stack([xs, ys], 1), H, W)


def _preencher(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    n = len(poly)
    for y in range(H):
        xs = []
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            a, b = int(np.ceil(xs[i])), int(np.floor(xs[i + 1]))
            if b >= 0 and a < W:
                mask[y, max(a, 0):min(b + 1, W)] = True
    return mask


def semente_de(mask: np.ndarray) -> tuple[int, int] | None:
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    if mask[cy, cx]:
        return cy, cx
    d = (ys - cy) ** 2 + (xs - cx) ** 2
    i = int(np.argmin(d))
    return int(ys[i]), int(xs[i])


def isolar(rgb: np.ndarray, semente_mask: np.ndarray) -> np.ndarray | None:
    s = semente_de(semente_mask)
    return None if s is None else contorno_por_raios(rgb, semente_mask, s)


def demo() -> None:
    H = W = 220
    yy, xx = np.mgrid[0:H, 0:W]

    rgb = np.full((H, W, 3), 208, np.uint8)
    rgb[170:] = 45
    rgb[:26] = 245
    lingua = (((yy - 110) / 62.0) ** 2 + ((xx - 110) / 44.0) ** 2) < 1
    rgb[lingua] = [198, 96, 102]
    rgb = (rgb + np.random.default_rng(0).integers(-6, 7, rgb.shape)).clip(0, 255).astype(np.uint8)

    frag = np.zeros((H, W), bool)
    frag[104:116, 104:116] = True

    m = isolar(rgb, frag)
    assert m is not None
    iou = (m & lingua).sum() / (m | lingua).sum()
    cob = (m & lingua).sum() / lingua.sum()
    print(f"  fragmento de {frag.mean():.4f} da imagem -> IoU {iou:.3f}, cobre {cob:.3f} da lingua")
    assert iou > 0.75, iou
    assert cob > 0.85, cob

    from hality.sam_segmenter import _componentes
    assert len(_componentes(m)) == 1, "contorno saiu fragmentado"
    assert len(_componentes(~m)) == 1, "sobrou buraco dentro da mascara"

    assert m[180:].mean() < 0.05, "vazou para a barba"
    assert m[:20].mean() < 0.05, "vazou para o dente"

    print("ok - contorno por raios (unico, fechado, sem vazar)")


if __name__ == "__main__":
    demo()
