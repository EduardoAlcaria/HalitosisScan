from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "sam", "sam_vit_b.pth")

N_POS = 6
N_NEG = 8
LADO = 640


def _componentes(mask: np.ndarray) -> list[np.ndarray]:
    visto = np.zeros_like(mask, bool)
    saida = []
    H, W = mask.shape
    for i0 in range(H):
        for j0 in range(W):
            if not mask[i0, j0] or visto[i0, j0]:
                continue
            pilha, pix = [(i0, j0)], []
            visto[i0, j0] = True
            while pilha:
                i, j = pilha.pop()
                pix.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and not visto[a, b]:
                        visto[a, b] = True
                        pilha.append((a, b))
            saida.append(np.array(pix))
    saida.sort(key=len, reverse=True)
    return saida


def _erode(mask: np.ndarray, n: int = 2) -> np.ndarray:
    m = mask.copy()
    for _ in range(n):
        e = m.copy()
        e[1:] &= m[:-1]
        e[:-1] &= m[1:]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


def sementes_por_cor(rgb: np.ndarray) -> np.ndarray | None:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    tom = (h < 25) | (h > 225)
    plausivel = tom & (s > 45) & (s < 210) & (v > 55) & (v < 245)
    if plausivel.sum() < 200:
        return None
    H, W = plausivel.shape
    yy, xx = np.mgrid[0:H, 0:W]
    perto = ((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2 < 0.85
    comps = _componentes(plausivel & perto)
    if not comps:
        return None
    maior = comps[0]
    out = np.zeros((H, W), bool)
    out[maior[:, 0], maior[:, 1]] = True
    return out if out.mean() > 0.01 else None


def gerar_prompts(rgb: np.ndarray, semente: np.ndarray | None,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    H, W = rgb.shape[:2]
    if semente is None or semente.sum() < 40:
        semente = sementes_por_cor(rgb)
    if semente is None or semente.sum() < 40:
        return None

    interior = _erode(semente, 2)
    if interior.sum() < 10:
        interior = semente
    ys, xs = np.where(interior)
    idx = rng.choice(len(ys), size=min(N_POS, len(ys)), replace=False)
    pos = np.stack([xs[idx], ys[idx]], 1)

    neg = _negativos(rgb, semente, rng)
    pontos = np.vstack([pos, neg]) if len(neg) else pos
    rotulos = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    return pontos.astype(int), rotulos


def _negativos(rgb: np.ndarray, semente: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    H, W = rgb.shape[:2]
    lab_L = np.asarray(Image.fromarray(rgb).convert("L"), np.float32) * 100.0 / 255.0
    perto = _erode(~semente, 0)
    anel = np.zeros_like(semente)
    d = semente.copy()
    for _ in range(max(3, int(min(H, W) * 0.08))):
        e = d.copy()
        e[1:] |= d[:-1]; e[:-1] |= d[1:]
        e[:, 1:] |= d[:, :-1]; e[:, :-1] |= d[:, 1:]
        d = e
    anel = d & ~semente & perto

    escolhidos = []
    for sel, quantos in ((anel & (lab_L < 28), 3),      # cavidade
                         (anel & (lab_L > 72), 2)):
        ys, xs = np.where(sel)
        if len(ys) >= quantos:
            i = rng.choice(len(ys), size=quantos, replace=False)
            escolhidos.append(np.stack([xs[i], ys[i]], 1))

    m = 0.04
    cand = np.array([[m, m], [1 - m, m], [m, 1 - m], [1 - m, 1 - m],
                     [0.5, m], [0.5, 1 - m], [m, 0.5], [1 - m, 0.5]])
    bordas = np.stack([cand[:, 0] * W, cand[:, 1] * H], 1).astype(int)
    bordas = bordas[~semente[np.clip(bordas[:, 1], 0, H - 1),
                             np.clip(bordas[:, 0], 0, W - 1)]]
    escolhidos.append(bordas)
    return np.vstack(escolhidos)[:N_NEG] if escolhidos else np.zeros((0, 2), int)


class SamLingua:
    def __init__(self, ckpt: str = CKPT, tipo: str = "vit_b") -> None:
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[tipo](checkpoint=ckpt)
        sam.eval()
        torch.set_num_threads(os.cpu_count() or 4)
        self.pred = SamPredictor(sam)

    def isolar(self, rgb: np.ndarray, semente: np.ndarray | None = None,
               seed: int = 0) -> np.ndarray | None:
        rng = np.random.default_rng(seed)
        H0, W0 = rgb.shape[:2]
        esc = LADO / max(H0, W0)
        if esc < 1:
            peq = np.asarray(Image.fromarray(rgb).resize(
                (int(W0 * esc), int(H0 * esc)), Image.BICUBIC))
            sem = None if semente is None else np.asarray(Image.fromarray(
                (semente.astype(np.uint8) * 255)).resize(peq.shape[1::-1], Image.NEAREST)) > 127
        else:
            peq, sem = rgb, semente

        p = gerar_prompts(peq, sem, rng)
        if p is None:
            return None
        pontos, rotulos = p

        self.pred.set_image(peq)
        masks, scores, _ = self.pred.predict(point_coords=pontos, point_labels=rotulos,
                                             multimask_output=True)
        melhor, melhor_nota = None, -1e9
        for m, s in zip(masks, scores):
            a = m.mean()
            if not (0.05 < a < 0.85):
                continue
            nota = float(s) - 2.0 * abs(a - 0.40)
            if nota > melhor_nota:
                melhor, melhor_nota = m, nota
        if melhor is None:
            melhor = masks[int(np.argmax(scores))]
        if melhor.shape != (H0, W0):
            melhor = np.asarray(Image.fromarray((melhor.astype(np.uint8) * 255)).resize(
                (W0, H0), Image.NEAREST)) > 127
        return melhor


def demo() -> None:
    rng = np.random.default_rng(0)
    H = W = 200

    rgb = np.full((H, W, 3), 205, np.uint8)
    rgb[150:] = 40
    yy, xx = np.mgrid[0:H, 0:W]
    lingua = (((yy - 100) / 45.0) ** 2 + ((xx - 100) / 32.0) ** 2) < 1
    rgb[lingua] = [200, 95, 100]

    s = sementes_por_cor(rgb)
    assert s is not None, "nao achou a lingua sintetica"
    assert (s & lingua).sum() / s.sum() > 0.9, (s & lingua).sum() / s.sum()

    p = gerar_prompts(rgb, s, rng)
    assert p is not None
    pontos, rotulos = p
    pos = pontos[rotulos == 1]
    assert len(pos) > 0 and rotulos.sum() < len(rotulos), "faltou positivo ou negativo"
    assert lingua[pos[:, 1], pos[:, 0]].all(), "ponto positivo caiu fora da lingua"
    neg = pontos[rotulos == 0]
    assert not lingua[neg[:, 1], neg[:, 0]].any(), "ponto negativo caiu dentro da lingua"

    assert sementes_por_cor(np.full((H, W, 3), 128, np.uint8)) is None

    assert _erode(lingua, 2).sum() < lingua.sum()
    assert (_erode(lingua, 2) & ~lingua).sum() == 0

    print("ok - gerador de prompts (sementes, positivos dentro, negativos fora)")


if __name__ == "__main__":
    demo()
