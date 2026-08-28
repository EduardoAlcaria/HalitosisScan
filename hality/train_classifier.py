from __future__ import annotations

import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, roc_curve

from .data import tabela_mestra, dividir, Amostra
from .features import FEATURE_NAMES, extract
from .pipeline import AREA_PLAUSIVEL_MIN, Hality
from .segmenter import SIZE as SEG_SIZE, UNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG = os.path.join(ROOT, "models", "segmentador.pt")
SAIDA = os.path.join(ROOT, "models", "classificador.pkl")

FEAT_SIZE = 128


AVISO_TESTE = """
ATENCAO: o conjunto de teste ja foi aberto uma vez, antes da cascata de resgate por
realce de contraste ser adicionada ao segment(). O numero abaixo NAO e mais um holdout
limpo -- a mudanca no pipeline foi feita com conhecimento do teste. Para uma estimativa
sem essa marca, e preciso um conjunto novo ou uma reparticao com semente diferente.
"""


def carregar_segmentador() -> UNet:
    ck = torch.load(SEG, map_location="cpu", weights_only=True)
    m = UNet(w=ck["w"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def _uma(modelo: UNet, rgb_full: np.ndarray) -> np.ndarray:
    x = np.asarray(Image.fromarray(rgb_full).resize((SEG_SIZE, SEG_SIZE), Image.BICUBIC),
                   np.float32) / 255.0
    prob = torch.sigmoid(modelo(torch.from_numpy(x.transpose(2, 0, 1))[None]))[0, 0].numpy()
    m = Image.fromarray((prob > 0.5).astype(np.uint8) * 255).resize(
        (FEAT_SIZE, FEAT_SIZE), Image.NEAREST)
    return np.asarray(m) > 127


def prever_mascara(modelo: UNet, rgb_full: np.ndarray) -> np.ndarray:
    m = _uma(modelo, rgb_full)
    if m.mean() < AREA_PLAUSIVEL_MIN:
        resgate = _uma(modelo, Hality._realce(rgb_full))
        if resgate.mean() > m.mean():
            return resgate
    return m


def _matriz(amostras: list[Amostra], seg: UNet):
    X, y, ok = [], [], []
    for a in amostras:
        rgb_full = np.asarray(Image.open(a.foto).convert("RGB"))
        mask = prever_mascara(seg, rgb_full)
        rgb = np.asarray(Image.fromarray(rgb_full).resize((FEAT_SIZE, FEAT_SIZE), Image.BICUBIC))
        try:
            X.append(extract(rgb, mask))
            y.append(a.alvo)
            ok.append(a.pid)
        except ValueError:
            pass
    return np.array(X), np.array(y), ok


def main() -> None:
    seg = carregar_segmentador()
    part = dividir(tabela_mestra())

    dados = {}
    for nome in ("treino", "val", "teste"):
        X, y, ok = _matriz(part[nome], seg)
        cob = len(ok) / len(part[nome])
        dados[nome] = (X, y)
        print(f"{nome}: n={len(y)}/{len(part[nome])}  cobertura={cob:.3f}  "
              f"prevalencia={y.mean():.3f}", flush=True)

    Xtr, ytr = dados["treino"]
    Xva, yva = dados["val"]
    Xte, yte = dados["teste"]

    base = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    modelo = CalibratedClassifierCV(base, method="isotonic", cv=5)
    modelo.fit(Xtr, ytr)

    pva = modelo.predict_proba(Xva)[:, 1]
    print(f"\nvalidacao: AUC={roc_auc_score(yva, pva):.3f}", flush=True)

    fpr, tpr, ths = roc_curve(yva, pva)
    viaveis = [(t, s, 1 - f) for t, s, f in zip(ths, tpr, fpr) if s >= 0.85]
    limiar, sens_va, esp_va = max(viaveis, key=lambda r: r[2])
    print(f"limiar escolhido={limiar:.3f}  sens_val={sens_va:.3f} espec_val={esp_va:.3f}",
          flush=True)

    lo, hi = np.percentile(pva, [40, 60])
    print(f"faixa de abstencao=[{lo:.3f}, {hi:.3f}]", flush=True)

    pte = modelo.predict_proba(Xte)[:, 1]
    pred = (pte >= limiar).astype(int)
    auc = roc_auc_score(yte, pte)
    print("\n" + "=" * 58, flush=True)
    print("CONJUNTO DE TESTE", flush=True)
    print("=" * 58, flush=True)
    print(AVISO_TESTE, flush=True)
    print(f"n={len(yte)}  prevalencia={yte.mean():.3f}")
    print(f"AUC      = {auc:.3f}")
    print(f"F1 macro = {f1_score(yte, pred, average='macro'):.3f}")
    print(f"acuracia = {accuracy_score(yte, pred):.3f}")
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"sensibilidade = {tp / (tp + fn):.3f}   especificidade = {tn / (tn + fp):.3f}")
    print(confusion_matrix(yte, pred), " linhas: 0=sem indicio 1=indicio")

    cobre = (pte < lo) | (pte > hi)
    if cobre.sum() > 5 and len(np.unique(yte[cobre])) > 1:
        print(f"\ncom abstencao: cobertura={cobre.mean():.2f}  "
              f"AUC={roc_auc_score(yte[cobre], pte[cobre]):.3f}  "
              f"acuracia={accuracy_score(yte[cobre], (pte[cobre] >= limiar).astype(int)):.3f}")
    print("\nreferencias: baseline majoritario F1=0.368 | anamnese sem Q6 AUC=0.615")

    with open(SAIDA, "wb") as f:
        pickle.dump({"modelo": modelo, "features": FEATURE_NAMES, "limiar": float(limiar),
                     "abstencao": [float(lo), float(hi)], "auc_teste": float(auc)}, f)
    print(f"\nsalvo em {SAIDA}", flush=True)

    with open(os.path.join(ROOT, "models", "metricas.json"), "w", encoding="utf-8") as f:
        json.dump({"auc_teste": float(auc), "n_teste": int(len(yte)),
                   "limiar": float(limiar), "abstencao": [float(lo), float(hi)],
                   "iou_segmentador_val": float(
                       torch.load(SEG, map_location="cpu", weights_only=True)["iou_val"])},
                  f, indent=2)


if __name__ == "__main__":
    main()
