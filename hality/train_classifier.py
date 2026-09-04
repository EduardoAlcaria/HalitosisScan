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
from .segmentacao import Segmentador

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG = os.path.join(ROOT, "models", "segmentador.pt")
SAIDA = os.path.join(ROOT, "models", "classificador.pkl")

FEAT_SIZE = 128

SENS_MINIMA = 0.80


AVISO_TESTE = """
ATENCAO: o conjunto de teste ja foi aberto uma vez, antes da cascata de resgate por
realce de contraste ser adicionada ao segment(). O numero abaixo NAO e mais um holdout
limpo -- a mudanca no pipeline foi feita com conhecimento do teste. Para uma estimativa
sem essa marca, e preciso um conjunto novo ou uma reparticao com semente diferente.
"""


def _matriz(amostras: list[Amostra], seg: Segmentador):
    X, y, ok = [], [], []
    for a in amostras:
        rgb_full = np.asarray(Image.open(a.foto).convert("RGB"))
        mask = seg(rgb_full)
        rgb = np.asarray(Image.fromarray(rgb_full).resize((FEAT_SIZE, FEAT_SIZE), Image.BICUBIC))
        try:
            X.append(extract(rgb, mask))
            y.append(a.alvo)
            ok.append(a.pid)
        except ValueError:
            pass
    return np.array(X), np.array(y), ok


def main() -> None:
    seg = Segmentador()
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
    viaveis = [(t, sn, 1 - f) for t, sn, f in zip(ths, tpr, fpr) if sn >= SENS_MINIMA]
    if not viaveis:
        viaveis = [(t, sn, 1 - f) for t, sn, f in zip(ths, tpr, fpr)]
    limiar, sens_va, esp_va = max(viaveis, key=lambda r: r[1] + r[2] - 1)
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
                   "iou_segmentador_val": float(seg.iou_val)},
                  f, indent=2)


if __name__ == "__main__":
    main()
