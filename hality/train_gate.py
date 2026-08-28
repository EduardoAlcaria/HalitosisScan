from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(ROOT, "models", "gate.pkl")
CN = os.path.join(ROOT, "drive-download-20260819T014023Z-1-001",
                  "中医舌诊染苔数据")
COCO = os.path.join(ROOT, "data_ext", "val2017")

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def preparar(caminho: str, rng) -> torch.Tensor | None:
    try:
        a = np.asarray(Image.open(caminho).convert("RGB"))
    except Exception:
        return None
    H, W = a.shape[:2]
    if min(H, W) < 64:
        return None
    lado = int(min(H, W) * rng.uniform(0.5, 1.0))
    y = rng.integers(0, H - lado + 1)
    x = rng.integers(0, W - lado + 1)
    crop = Image.fromarray(a[y:y + lado, x:x + lado])
    eff = int(rng.integers(96, 225))
    crop = crop.resize((eff, eff), Image.BICUBIC).resize((224, 224), Image.BICUBIC)
    t = np.asarray(crop, np.float32) / 255.0
    return torch.from_numpy(((t - MEAN) / STD).transpose(2, 0, 1))


def coletar() -> list[tuple[str, int, str, str]]:
    itens = []
    for f in sorted(glob.glob(os.path.join(ROOT, "Hality-Project-main", "data",
                                           "Classificacao", "*"))):
        pid = os.path.splitext(os.path.basename(f))[0].lower()
        itens.append((f, 1, "own_" + pid, "propria"))
    for i, f in enumerate(sorted(glob.glob(os.path.join(CN, "*", "*")))):
        itens.append((f, 1, "cn_%d" % i, "chinesa"))
    for i, f in enumerate(sorted(glob.glob(os.path.join(COCO, "*.jpg")))):
        itens.append((f, 0, "coco_%d" % i, "coco"))
    return itens


def main() -> None:
    rng = np.random.default_rng(0)
    itens = coletar()
    y = np.array([i[1] for i in itens])
    src = np.array([i[3] for i in itens])
    grupo = np.array([i[2] for i in itens])
    print("positivos: %d proprias + %d chinesas | negativos: %d COCO"
          % ((src == "propria").sum(), (src == "chinesa").sum(), (src == "coco").sum()),
          flush=True)

    dims = []
    for f, *_ in itens:
        try:
            w, h = Image.open(f).size
        except Exception:
            w = h = 0
        dims.append([w, h, w * h, w / max(h, 1)])
    dims = np.array(dims, float)
    from sklearn.ensemble import HistGradientBoostingClassifier
    pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0),
                           dims, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    auc_dim = roc_auc_score(y, pr)
    print("confundidor por dimensoes nativas: AUC=%.4f" % auc_dim, flush=True)
    print("  (alto e esperado -- foto de boca em close tem formato proprio. O que importa",
          flush=True)
    print("   e se sobrevive ao tratamento identico abaixo, medido na imagem final.)",
          flush=True)

    mdl = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    torch.set_num_threads(os.cpu_count() or 4)
    E, manter, nitidez = [], [], []
    lote: list[torch.Tensor] = []
    with torch.no_grad():
        for k, (f, *_) in enumerate(itens):
            t = preparar(f, rng)
            if t is None:
                continue
            lum = (t.numpy().transpose(1, 2, 0) * STD + MEAN).mean(2) * 255
            lap = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1]
                         - lum[1:-1, :-2] - lum[1:-1, 2:])
            nitidez.append(lap.var())
            lote.append(t)
            manter.append(k)
            if len(lote) == 32:
                o = mdl(pixel_values=torch.stack(lote)).last_hidden_state
                E.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy())
                lote = []
            if k % 800 == 0:
                print("  %d/%d" % (k, len(itens)), flush=True)
        if lote:
            o = mdl(pixel_values=torch.stack(lote)).last_hidden_state
            E.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy())

    E = np.vstack(E)
    manter = np.array(manter)
    y, src, grupo = y[manter], src[manter], grupo[manter]
    nitidez = np.array(nitidez).reshape(-1, 1)
    print("embeddings: %s" % (E.shape,), flush=True)

    pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0),
                           nitidez, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    print("confundidor por NITIDEZ da imagem final: AUC=%.4f  (alvo: perto de 0.5)"
          % roc_auc_score(y, pr), flush=True)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.01, max_iter=4000, class_weight="balanced"))
    p = cross_val_predict(clf, E, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                          groups=grupo, method="predict_proba")[:, 1]
    print("\n=== GATE (CV agrupada por foto de origem) ===", flush=True)
    print("AUC = %.4f" % roc_auc_score(y, p))
    print(confusion_matrix(y, (p > 0.5).astype(int)), " linhas: 0=nao-lingua 1=lingua")

    proprias = p[(src == "propria")]
    limiar = float(np.quantile(proprias, 0.02))
    print("\nlimiar para 98%% de recall nas fotos PROPRIAS: %.4f" % limiar)
    for s in ("propria", "chinesa"):
        k = src == s
        print("  recall %-8s = %6.2f%%  (n=%d)" % (s, 100 * (p[k] >= limiar).mean(), k.sum()))
    k = src == "coco"
    print("  falso-aceite coco = %6.2f%%  (n=%d)" % (100 * (p[k] >= limiar).mean(), k.sum()))

    clf.fit(E, y)
    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "wb") as f:
        pickle.dump({"modelo": clf, "limiar": limiar,
                     "auc_cv": float(roc_auc_score(y, p)),
                     "recall_proprias": float((p[src == "propria"] >= limiar).mean()),
                     "falso_aceite_coco": float((p[src == "coco"] >= limiar).mean())}, f)
    print("\nsalvo em %s" % SAIDA, flush=True)


if __name__ == "__main__":
    main()
