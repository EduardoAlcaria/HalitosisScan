from __future__ import annotations

import io
import os
import pickle
from dataclasses import dataclass, asdict

import numpy as np
import torch
from PIL import Image, ImageOps

from .features import extract
from .segmenter import SIZE as SEG_SIZE, UNet

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(ROOT, "models")
FEAT_SIZE = 128
LADO_MAX = 1024

NITIDEZ_MIN = 12.0
CLARO_MAX = 0.35
ESCURO_MAX = 0.55
AREA_MIN = 0.12
AREA_PLAUSIVEL_MIN = 0.20
AREA_FRAGMENTO_MIN = 0.55

N_PASSAGENS = 5
K_ABSTENCAO = 2.0


@dataclass
class Resultado:
    veredito: str
    motivo: str
    probabilidade: float | None = None
    area_lingua: float | None = None
    nitidez: float | None = None
    confianca_lingua: float | None = None
    dispersao: float | None = None
    passagens: int | None = None

    def dict(self) -> dict:
        return asdict(self)


class Hality:
    def __init__(self) -> None:
        ck = torch.load(os.path.join(MODELOS, "segmentador.pt"),
                        map_location="cpu", weights_only=True)
        self.seg = UNet(w=ck["w"])
        self.seg.load_state_dict(ck["state_dict"])
        self.seg.eval()
        self.iou_seg = ck["iou_val"]

        with open(os.path.join(MODELOS, "classificador.pkl"), "rb") as f:
            art = pickle.load(f)
        self.clf = art["modelo"]
        self.limiar = art["limiar"]
        self.auc_teste = art["auc_teste"]

        self.ruido = float(art.get("ruido", 0.0))
        if self.ruido > 0:
            meia = K_ABSTENCAO * self.ruido
            self.abst_lo, self.abst_hi = self.limiar - meia, self.limiar + meia
        else:
            self.abst_lo, self.abst_hi = art["abstencao"]

        self.gate = None
        caminho_gate = os.path.join(MODELOS, "gate.pkl")
        if os.path.exists(caminho_gate):
            with open(caminho_gate, "rb") as f:
                g = pickle.load(f)
            self.gate = g["modelo"]
            self.gate_limiar = g["limiar"]
            self._dino = None

    def _embedding(self, rgb: np.ndarray) -> np.ndarray:
        if self._dino is None:
            from transformers import AutoModel
            self._dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        im = Image.fromarray(rgb)
        lado = min(im.size)
        esq, topo = (im.width - lado) // 2, (im.height - lado) // 2
        im = im.crop((esq, topo, esq + lado, topo + lado)).resize((224, 224), Image.BICUBIC)
        t = np.asarray(im, np.float32) / 255.0
        media = np.array([.485, .456, .406], np.float32)
        desvio = np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(((t - media) / desvio).transpose(2, 0, 1))[None]
        with torch.no_grad():
            o = self._dino(pixel_values=x).last_hidden_state
        return torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy()

    def tongue_gate(self, rgb: np.ndarray) -> tuple[bool, float]:
        if self.gate is None:
            return True, 1.0
        p = float(self.gate.predict_proba(self._embedding(rgb))[0, 1])
        return p >= self.gate_limiar, p

    @staticmethod
    def normalize(raw: bytes) -> np.ndarray:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > LADO_MAX:
            s = LADO_MAX / max(im.size)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                           Image.BICUBIC)
        return np.asarray(im)

    @staticmethod
    def image_quality(rgb: np.ndarray) -> tuple[bool, str, float]:
        lum = rgb.mean(2).astype(np.float32)
        lap = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1]
                     - lum[1:-1, :-2] - lum[1:-1, 2:])
        nitidez = float(lap.var())
        if (lum > 250).mean() > CLARO_MAX:
            return False, "Foto muito clara. Evite luz direta ou flash proximo.", nitidez
        if (lum < 30).mean() > ESCURO_MAX or lum.mean() < 40:
            return False, "Foto muito escura. Procure um ambiente mais iluminado.", nitidez
        if nitidez < NITIDEZ_MIN:
            return False, "Foto tremida ou fora de foco. Refaca segurando firme.", nitidez
        return True, "", nitidez

    @staticmethod
    def _realce(rgb: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
        o = rgb.astype(np.float32).copy()
        for c in range(3):
            a, b = np.percentile(o[..., c], [lo, hi])
            if b > a:
                o[..., c] = np.clip((o[..., c] - a) * 255 / (b - a), 0, 255)
        return o.astype(np.uint8)

    @torch.no_grad()
    def _segment_uma(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(Image.fromarray(rgb).resize((SEG_SIZE, SEG_SIZE), Image.BICUBIC),
                       np.float32) / 255.0
        prob = torch.sigmoid(self.seg(torch.from_numpy(x.transpose(2, 0, 1))[None]))
        m = (prob[0, 0].numpy() > 0.5).astype(np.uint8) * 255
        return np.asarray(Image.fromarray(m).resize((FEAT_SIZE, FEAT_SIZE),
                                                    Image.NEAREST)) > 127

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        m = self._segment_uma(rgb)
        if m.mean() < AREA_PLAUSIVEL_MIN:
            resgate = self._segment_uma(self._realce(rgb))
            if resgate.mean() > m.mean():
                return resgate
        return m

    @staticmethod
    def mask_sanity(mask: np.ndarray) -> tuple[bool, str]:
        area = mask.mean()
        if area < AREA_MIN:
            return False, "Nao identifiquei uma lingua. Aproxime a camera e mostre a lingua."
        if _maior_componente(mask) / mask.sum() < AREA_FRAGMENTO_MIN:
            return False, "Nao consegui delimitar a lingua. Refaca a foto."
        return True, ""

    def _variantes(self, rgb: np.ndarray):
        rng = np.random.default_rng(12345)
        yield rgb
        for _ in range(N_PASSAGENS - 1):
            im = Image.fromarray(np.ascontiguousarray(rgb))
            W, H = im.size
            im = im.rotate(rng.uniform(-5, 5), resample=Image.BICUBIC)
            im = im.crop((int(W * rng.uniform(0, .07)), int(H * rng.uniform(0, .07)),
                          W - int(W * rng.uniform(0, .07)), H - int(H * rng.uniform(0, .07))))
            v = np.asarray(im)
            yield v[:, ::-1] if rng.random() < 0.5 else v

    def _uma_passagem(self, rgb: np.ndarray, nitidez: float) -> Resultado:
        ok, motivo, nit = self.image_quality(rgb)
        if not ok:
            return Resultado("rejeitado", motivo, nitidez=nit)
        eh, conf = self.tongue_gate(rgb)
        if not eh:
            return Resultado("rejeitado", "Nao identifiquei uma lingua nesta foto. "
                             "Mostre a lingua para fora, de frente para a camera.",
                             nitidez=nit, confianca_lingua=round(conf, 4))
        m = self.segment(rgb)
        ok, motivo = self.mask_sanity(m)
        if not ok:
            return Resultado("rejeitado", motivo, area_lingua=float(m.mean()), nitidez=nit,
                             confianca_lingua=round(conf, 4))
        peq = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (FEAT_SIZE, FEAT_SIZE), Image.BICUBIC))
        try:
            v = extract(peq, m)
        except ValueError:
            return Resultado("rejeitado", "Regiao da lingua pequena demais. Aproxime a camera.",
                             area_lingua=float(m.mean()), nitidez=nit)
        p = float(np.clip(self.clf.predict_proba(v[None])[0, 1], 0.02, 0.98))
        ver = ("inconclusivo" if self.abst_lo <= p <= self.abst_hi
               else ("indicio" if p >= self.limiar else "sem_indicio"))
        return Resultado(ver, "", probabilidade=p, area_lingua=float(m.mean()),
                         nitidez=nit, confianca_lingua=round(conf, 4))

    MOTIVOS = {
        "indicio": "Indicio compativel com halitose. Procure um dentista para "
                   "avaliacao. Isto nao e um diagnostico.",
        "sem_indicio": "Sem indicio na imagem. Isto nao descarta halitose; em caso de "
                       "duvida, procure um dentista.",
        "inconclusivo": "Nao ha base suficiente para uma estimativa. Refaca a foto ou "
                        "procure avaliacao odontologica.",
    }

    def analisar(self, raw: bytes) -> Resultado:
        try:
            rgb = self.normalize(raw)
        except Exception:
            return Resultado("rejeitado", "Nao consegui ler a imagem. Envie JPEG, PNG, BMP ou HEIC.")

        _, _, nitidez = self.image_quality(rgb)
        passagens = [self._uma_passagem(v, nitidez) for v in self._variantes(rgb)]

        votos: dict[str, int] = {}
        for r in passagens:
            votos[r.veredito] = votos.get(r.veredito, 0) + 1
        vencedor, n = max(votos.items(), key=lambda kv: kv[1])

        probs = [r.probabilidade for r in passagens if r.probabilidade is not None]
        p = float(np.mean(probs)) if probs else None
        disp = float(np.std(probs)) if len(probs) > 1 else 0.0
        areas = [r.area_lingua for r in passagens if r.area_lingua is not None]

        comum = dict(probabilidade=None if p is None else round(p, 4),
                     area_lingua=round(float(np.mean(areas)), 4) if areas else None,
                     nitidez=round(nitidez, 1), dispersao=round(disp, 4),
                     passagens=len(passagens),
                     confianca_lingua=next((r.confianca_lingua for r in passagens
                                            if r.confianca_lingua is not None), None))

        if n <= len(passagens) // 2:
            return Resultado("inconclusivo",
                             "O resultado nao se manteve estavel entre as analises desta "
                             "foto. Refaca a foto com a lingua bem visivel e boa luz.",
                             **comum)

        if vencedor == "rejeitado":
            motivo = next(r.motivo for r in passagens if r.veredito == "rejeitado")
            return Resultado("rejeitado", motivo, **comum)
        return Resultado(vencedor, self.MOTIVOS[vencedor], **comum)


def _maior_componente(mask: np.ndarray) -> int:
    visto = np.zeros_like(mask, bool)
    maior = 0
    H, W = mask.shape
    for i0 in range(H):
        for j0 in range(W):
            if not mask[i0, j0] or visto[i0, j0]:
                continue
            pilha, tam = [(i0, j0)], 0
            visto[i0, j0] = True
            while pilha:
                i, j = pilha.pop()
                tam += 1
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and not visto[a, b]:
                        visto[a, b] = True
                        pilha.append((a, b))
            maior = max(maior, tam)
    return maior


def demo() -> None:
    rng = np.random.default_rng(0)

    ruido = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    assert Hality.image_quality(ruido)[0]
    liso = np.full((200, 200, 3), 128, np.uint8)
    ok, motivo, _ = Hality.image_quality(liso)
    assert not ok and "tremida" in motivo

    branco = np.full((200, 200, 3), 255, np.uint8)
    branco[:10] = rng.integers(0, 256, (10, 200, 3))
    ok, motivo, _ = Hality.image_quality(branco)
    assert not ok and "clara" in motivo, motivo

    escura = (ruido * 0.12).astype(np.uint8)
    ok, motivo, _ = Hality.image_quality(escura)
    assert not ok and "escura" in motivo, motivo

    m = np.zeros((128, 128), bool); m[30:100, 30:100] = True
    assert Hality.mask_sanity(m)[0]
    assert not Hality.mask_sanity(np.zeros((128, 128), bool))[0]
    dois = np.zeros((128, 128), bool); dois[5:55, 5:55] = True; dois[70:120, 70:120] = True
    assert dois.mean() > AREA_MIN, dois.mean()
    ok, motivo = Hality.mask_sanity(dois)
    assert not ok and "delimitar" in motivo, motivo

    grande = np.zeros((128, 128), bool); grande[8:120, 8:120] = True
    assert grande.mean() > 0.6 and Hality.mask_sanity(grande)[0]

    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (2000, 1500, 3), dtype=np.uint8)).save(buf, "JPEG")
    out = Hality.normalize(buf.getvalue())
    assert max(out.shape[:2]) == LADO_MAX and out.shape[2] == 3

    print("ok - etapas puras do pipeline")


if __name__ == "__main__":
    demo()
