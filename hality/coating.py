from __future__ import annotations

import numpy as np
from PIL import Image

TAM = 128
MIN_PX = 200

COATING_NAMES = [
    "saburra_area",
    "saburra_L",
    "saburra_croma",
    "saburra_contraste_L",
    "saburra_post",
    "saburra_med",
    "saburra_ant",
    "saburra_post_ant",
    "saburra_heterogen",
    "saburra_espalhamento",
    "lingua_alongamento",
    "lingua_L_medio",
]


def _lab(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    m = x > 0.04045
    x[m] = ((x[m] + 0.055) / 1.055) ** 2.4
    x[~m] = x[~m] / 12.92
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        o = np.empty_like(t)
        m = t > 0.008856
        o[m] = np.cbrt(t[m])
        o[~m] = 7.787 * t[~m] + 16 / 116
        return o

    fx, fy, fz = f(X), f(Y), f(Z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def eixo_anatomico(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.where(mask)
    pts = np.stack([ys, xs], 1).astype(np.float32)
    pts -= pts.mean(0)
    cov = np.cov(pts.T)
    val, vec = np.linalg.eigh(cov)
    principal = vec[:, -1]
    alongamento = float(np.sqrt(val[-1] / max(val[0], 1e-6)))

    t = pts @ principal
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)

    secundario = vec[:, 0]
    w = np.abs(pts @ secundario)
    if w[t < 0.5].mean() > w[t > 0.5].mean():
        t = 1.0 - t
    return t, (ys, xs), alongamento


def segmentar_saburra(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    L = lab[ys, xs, 0]
    croma = np.sqrt(lab[ys, xs, 1] ** 2 + lab[ys, xs, 2] ** 2)
    X = np.stack([(L - L.mean()) / (L.std() + 1e-6),
                  (croma - croma.mean()) / (croma.std() + 1e-6)], 1)

    escore = X[:, 0] - X[:, 1]
    c = np.stack([X[escore.argmin()], X[escore.argmax()]])
    for _ in range(25):
        d = ((X[:, None] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for k in (0, 1):
            if (a == k).any():
                c[k] = X[a == k].mean(0)
    alvo = int(np.argmax(c[:, 0] - c[:, 1]))
    out = np.zeros_like(mask)
    out[ys[a == alvo], xs[a == alvo]] = True
    return out


def extract_coating(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if rgb.shape[:2] != (TAM, TAM):
        rgb = np.asarray(Image.fromarray(rgb).convert("RGB").resize((TAM, TAM), Image.BICUBIC))
    if mask.shape != (TAM, TAM):
        mask = np.asarray(Image.fromarray((mask.astype(np.uint8) * 255)).resize(
            (TAM, TAM), Image.NEAREST)) > 127
    if mask.sum() < MIN_PX:
        raise ValueError("mascara pequena demais para analise de saburra")

    lab = _lab(rgb)
    sab = segmentar_saburra(lab, mask)
    t, (ys, xs), alongamento = eixo_anatomico(mask)
    e_sab = sab[ys, xs]

    L_sab = lab[ys, xs, 0][e_sab]
    L_pap = lab[ys, xs, 0][~e_sab]
    croma = np.sqrt(lab[ys, xs, 1] ** 2 + lab[ys, xs, 2] ** 2)

    def cobertura(lo, hi):
        faixa = (t >= lo) & (t < hi)
        return float(e_sab[faixa].mean()) if faixa.sum() > 30 else 0.0

    ant, med, post = cobertura(0, 1 / 3), cobertura(1 / 3, 2 / 3), cobertura(2 / 3, 1.01)
    finas = [cobertura(i / 8, (i + 1) / 8) for i in range(8)]

    f = [
        float(e_sab.mean()),
        float(L_sab.mean()) if e_sab.any() else 0.0,
        float(croma[e_sab].mean()) if e_sab.any() else 0.0,
        float(L_sab.mean() - L_pap.mean()) if e_sab.any() and (~e_sab).any() else 0.0,
        post, med, ant,
        float(post / max(ant, 1e-3)),
        float(np.std(finas)),
        float(t[e_sab].std()) if e_sab.any() else 0.0,
        alongamento,
        float(lab[ys, xs, 0].mean()),
    ]
    v = np.nan_to_num(np.array(f, float), posinf=0.0, neginf=0.0)
    assert len(v) == len(COATING_NAMES)
    return v


def demo() -> None:
    rng = np.random.default_rng(0)

    yy, xx = np.mgrid[0:TAM, 0:TAM]
    mask = (((yy - 64) / 46.0) ** 2 + ((xx - 64) / 26.0) ** 2) < 1
    rgb = np.zeros((TAM, TAM, 3), np.uint8)
    rgb[mask] = [190, 110, 110]
    faixa = mask & (yy > 78)
    rgb[faixa] = [232, 226, 214]

    v = extract_coating(rgb, mask)
    d = dict(zip(COATING_NAMES, v))

    assert 0.05 < d["saburra_area"] < 0.6, d["saburra_area"]
    assert d["saburra_contraste_L"] > 10, d["saburra_contraste_L"]
    assert d["lingua_alongamento"] > 1.3, d["lingua_alongamento"]

    assert d["saburra_post"] > d["saburra_ant"], (d["saburra_post"], d["saburra_ant"])

    girada = np.rot90(rgb).copy(), np.rot90(mask).copy()
    g = dict(zip(COATING_NAMES, extract_coating(*girada)))
    assert abs(g["saburra_area"] - d["saburra_area"]) < 0.06
    assert g["saburra_post"] > g["saburra_ant"], "orientacao quebrou ao girar a imagem"

    limpa = np.zeros((TAM, TAM, 3), np.uint8)
    limpa[mask] = [190, 110, 110]
    limpa[mask] += rng.integers(0, 12, (mask.sum(), 3), dtype=np.uint8)
    assert dict(zip(COATING_NAMES, extract_coating(limpa, mask)))["saburra_contraste_L"] \
        < d["saburra_contraste_L"]

    print(f"ok - {len(COATING_NAMES)} features de saburra, orientacao invariante a rotacao")


if __name__ == "__main__":
    demo()
