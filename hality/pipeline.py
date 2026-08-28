"""Pipeline de inferencia: bytes de imagem -> veredito.

Etapas, na ordem. Cada rejeicao e uma resposta bem-sucedida com motivo especifico --
"foto tremida" ajuda o usuario, "erro" nao.

  1 normalize        decodifica, corrige EXIF, reamostra preservando proporcao
  2 image_quality    exposicao e nitidez          -> rejeita
  3 tongue_gate      Modelo A: ha uma lingua?     -> rejeita
  4 segment          mascara da lingua
  5 mask_sanity      mascara vazia ou fragmentada -> rejeita
  6 extract          features de cor e textura
  7 predict          probabilidade calibrada
  8 decide           faixa de abstencao           -> inconclusivo

LIMITE CONHECIDO DO GATE: seus negativos sao fotos do COCO -- cenas arbitrarias. Isso
cobre a captura acidental (bolso, chao, teto), que e separacao facil. O caso dificil,
rosto de boca fechada ou sem a lingua para fora, NAO existe no conjunto de treino e
nao foi testado. Antes de abrir ao publico, colete esses negativos.
"""
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

try:                      # HEIC e comum em foto de iPhone
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(ROOT, "models")
FEAT_SIZE = 128
LADO_MAX = 1024

# Limiares de qualidade, calibrados sobre a distribuicao das fotos proprias.
NITIDEZ_MIN = 12.0
CLARO_MAX = 0.35          # fracao de pixels estourados no branco
ESCURO_MAX = 0.55         # fracao de pixels colados no preto
AREA_MIN = 0.06
AREA_FRAGMENTO_MIN = 0.55  # maior componente / area total da mascara


@dataclass
class Resultado:
    veredito: str                    # indicio | sem_indicio | inconclusivo | rejeitado
    motivo: str
    probabilidade: float | None = None
    area_lingua: float | None = None
    nitidez: float | None = None
    confianca_lingua: float | None = None

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

        # pickle e seguro aqui: o arquivo e produzido por hality/train_classifier.py
        # dentro do repositorio, nunca recebido de fora. E o formato padrao de
        # persistencia de estimadores scikit-learn. Se algum dia o artefato passar a vir
        # de fonte externa, trocar por ONNX ou skops antes.
        with open(os.path.join(MODELOS, "classificador.pkl"), "rb") as f:
            art = pickle.load(f)
        self.clf = art["modelo"]
        self.limiar = art["limiar"]
        self.abst_lo, self.abst_hi = art["abstencao"]
        self.auc_teste = art["auc_teste"]

        # Modelo A. Opcional: o pipeline roda sem ele, com protecao mais fraca.
        self.gate = None
        caminho_gate = os.path.join(MODELOS, "gate.pkl")
        if os.path.exists(caminho_gate):
            with open(caminho_gate, "rb") as f:      # mesmo raciocinio de seguranca acima
                g = pickle.load(f)
            self.gate = g["modelo"]
            self.gate_limiar = g["limiar"]
            self._dino = None                        # carregado no primeiro uso

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

    # ---- 3 ----
    def tongue_gate(self, rgb: np.ndarray) -> tuple[bool, float]:
        if self.gate is None:
            return True, 1.0
        p = float(self.gate.predict_proba(self._embedding(rgb))[0, 1])
        return p >= self.gate_limiar, p

    # ---- 1 ----
    @staticmethod
    def normalize(raw: bytes) -> np.ndarray:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > LADO_MAX:                      # preserva a proporcao
            s = LADO_MAX / max(im.size)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                           Image.BICUBIC)
        return np.asarray(im)

    # ---- 2 ----
    @staticmethod
    def image_quality(rgb: np.ndarray) -> tuple[bool, str, float]:
        lum = rgb.mean(2).astype(np.float32)
        lap = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1]
                     - lum[1:-1, :-2] - lum[1:-1, 2:])
        nitidez = float(lap.var())
        # Exposicao ANTES de nitidez: uma foto escura tem pouco contraste e reprovaria
        # na nitidez primeiro, devolvendo "tremida" para quem so precisa de mais luz.
        if (lum > 250).mean() > CLARO_MAX:
            return False, "Foto muito clara. Evite luz direta ou flash proximo.", nitidez
        if (lum < 30).mean() > ESCURO_MAX or lum.mean() < 40:
            return False, "Foto muito escura. Procure um ambiente mais iluminado.", nitidez
        if nitidez < NITIDEZ_MIN:
            return False, "Foto tremida ou fora de foco. Refaca segurando firme.", nitidez
        return True, "", nitidez

    # ---- 3 ----
    @torch.no_grad()
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(Image.fromarray(rgb).resize((SEG_SIZE, SEG_SIZE), Image.BICUBIC),
                       np.float32) / 255.0
        prob = torch.sigmoid(self.seg(torch.from_numpy(x.transpose(2, 0, 1))[None]))
        m = (prob[0, 0].numpy() > 0.5).astype(np.uint8) * 255
        return np.asarray(Image.fromarray(m).resize((FEAT_SIZE, FEAT_SIZE),
                                                    Image.NEAREST)) > 127

    # ---- 4 ----
    @staticmethod
    def mask_sanity(mask: np.ndarray) -> tuple[bool, str]:
        area = mask.mean()
        if area < AREA_MIN:
            return False, "Nao identifiquei uma lingua. Aproxime a camera e mostre a lingua."
        # sem limite superior de area: mascara cobrindo 70% do quadro e close-up
        # legitimo, e esta entre as melhores imagens do conjunto (ARQUITETURA 6.2)
        if _maior_componente(mask) / mask.sum() < AREA_FRAGMENTO_MIN:
            return False, "Nao consegui delimitar a lingua. Refaca a foto."
        return True, ""

    def analisar(self, raw: bytes) -> Resultado:
        try:
            rgb = self.normalize(raw)
        except Exception:
            return Resultado("rejeitado", "Nao consegui ler a imagem. Envie JPEG, PNG, BMP ou HEIC.")

        ok, motivo, nitidez = self.image_quality(rgb)
        if not ok:
            return Resultado("rejeitado", motivo, nitidez=nitidez)

        eh_lingua, conf = self.tongue_gate(rgb)
        if not eh_lingua:
            return Resultado("rejeitado",
                             "Nao identifiquei uma lingua nesta foto. "
                             "Mostre a lingua para fora, de frente para a camera.",
                             nitidez=nitidez, confianca_lingua=round(conf, 4))

        mask = self.segment(rgb)
        ok, motivo = self.mask_sanity(mask)
        if not ok:
            return Resultado("rejeitado", motivo, area_lingua=float(mask.mean()), nitidez=nitidez)

        pequena = np.asarray(Image.fromarray(rgb).resize((FEAT_SIZE, FEAT_SIZE), Image.BICUBIC))
        try:
            v = extract(pequena, mask)
        except ValueError:
            return Resultado("rejeitado", "Regiao da lingua pequena demais. Aproxime a camera.",
                             area_lingua=float(mask.mean()), nitidez=nitidez)

        # A regressao isotonica satura em 0 e 1 exatos quando calibrada com poucos
        # dados. Certeza absoluta a partir de 213 amostras de treino nao e defensavel,
        # e em contexto de saude e pior ainda -- o recorte mantem a saida honesta.
        p = float(np.clip(self.clf.predict_proba(v[None])[0, 1], 0.02, 0.98))
        comum = dict(probabilidade=round(p, 4), area_lingua=round(float(mask.mean()), 4),
                     nitidez=round(nitidez, 1), confianca_lingua=round(conf, 4))

        if self.abst_lo <= p <= self.abst_hi:
            return Resultado("inconclusivo",
                             "Nao ha base suficiente para uma estimativa. "
                             "Refaca a foto ou procure avaliacao odontologica.", **comum)
        if p >= self.limiar:
            return Resultado("indicio",
                             "Indicio compativel com halitose. Procure um dentista "
                             "para avaliacao. Isto nao e um diagnostico.", **comum)
        return Resultado("sem_indicio",
                         "Sem indicio na imagem. Isto nao descarta halitose; "
                         "em caso de duvida, procure um dentista.", **comum)


def _maior_componente(mask: np.ndarray) -> int:
    """Maior componente conexa, por varredura iterativa (sem dependencia extra)."""
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
    """Auto-teste das etapas puras: `python -m hality.pipeline`."""
    rng = np.random.default_rng(0)

    # nitidez: ruido e nitido, borrao uniforme nao
    ruido = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    assert Hality.image_quality(ruido)[0]
    liso = np.full((200, 200, 3), 128, np.uint8)
    ok, motivo, _ = Hality.image_quality(liso)
    assert not ok and "tremida" in motivo

    # exposicao, e o MOTIVO tem de ser o certo: escura reprova por luz, nao por tremor
    branco = np.full((200, 200, 3), 255, np.uint8)
    branco[:10] = rng.integers(0, 256, (10, 200, 3))     # um pouco de textura
    ok, motivo, _ = Hality.image_quality(branco)
    assert not ok and "clara" in motivo, motivo

    escura = (ruido * 0.12).astype(np.uint8)             # nitida, porem sem luz
    ok, motivo, _ = Hality.image_quality(escura)
    assert not ok and "escura" in motivo, motivo

    # mask_sanity: blob unico passa, area minuscula e dois blobs separados nao
    m = np.zeros((128, 128), bool); m[30:100, 30:100] = True
    assert Hality.mask_sanity(m)[0]
    assert not Hality.mask_sanity(np.zeros((128, 128), bool))[0]
    dois = np.zeros((128, 128), bool); dois[5:35, 5:35] = True; dois[90:120, 90:120] = True
    ok, motivo = Hality.mask_sanity(dois)
    assert not ok and "delimitar" in motivo

    # area alta NAO pode ser rejeitada (close-up legitimo)
    grande = np.zeros((128, 128), bool); grande[8:120, 8:120] = True
    assert grande.mean() > 0.6 and Hality.mask_sanity(grande)[0]

    # normalize aceita bytes e respeita o lado maximo
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (2000, 1500, 3), dtype=np.uint8)).save(buf, "JPEG")
    out = Hality.normalize(buf.getvalue())
    assert max(out.shape[:2]) == LADO_MAX and out.shape[2] == 3

    print("ok - etapas puras do pipeline")


if __name__ == "__main__":
    demo()
