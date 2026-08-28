"""Features do Modelo B.

Todas calculadas apenas sobre os pixels dentro da mascara da lingua.

Os NOMES e a ORDEM fazem parte do contrato: o modelo treinado depende deles.
Alterar a extracao invalida o artefato salvo -- por isso FEATURE_NAMES e gravado
junto com o modelo e conferido no carregamento.

Medicao que orienta o desenho (docs/ARQUITETURA.md secao 3): a feature dominante e
HSV_S_p10, o percentil 10 da saturacao, ou seja quao palida e a porcao mais palida da
lingua. Isso e a camada de saburra. Uma versao limiarizada da mesma ideia
("fracao de pixels com saturacao < 60") foi testada e vale AUC 0.553 sozinha, contra
0.799 do resto -- binarizar descarta a informacao que o percentil preserva.
Regra: medidas continuas por percentil, nunca contagens limiarizadas.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 128          # resolucao de trabalho da extracao
MIN_MASK_PX = 200   # abaixo disso a mascara nao sustenta estatistica

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
    """rgb uint8 [H,W,3], mask bool [H,W] -> vetor float64 [len(FEATURE_NAMES)]."""
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

    # ponta / meio / base: a saburra concentra no fundo do dorso
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
    """Auto-teste: roda com `python -m hality.features`."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), bool)
    mask[50:150, 80:220] = True

    v = extract(rgb, mask)
    assert v.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(v).all()

    # area_mascara reflete a fracao coberta (~0.233), com folga pela reamostragem
    area = v[FEATURE_NAMES.index("area_mascara")]
    assert 0.18 < area < 0.29, area

    # uma lingua uniformemente palida tem S_p10 menor que uma saturada
    pal = np.full((128, 128, 3), 230, np.uint8)
    sat = np.zeros((128, 128, 3), np.uint8)
    sat[..., 0] = 200
    m = np.ones((128, 128), bool)
    i = FEATURE_NAMES.index("HSV_S_p10")
    assert extract(pal, m)[i] < extract(sat, m)[i]

    # mascara pequena demais precisa falhar, nao devolver lixo
    try:
        extract(rgb, np.zeros((200, 300), bool))
        raise AssertionError("deveria ter levantado ValueError")
    except ValueError:
        pass

    # estabilidade: mesma entrada, mesma saida
    assert np.allclose(extract(rgb, mask), v)

    print(f"ok - {len(FEATURE_NAMES)} features")


if __name__ == "__main__":
    demo()
