from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .pipeline import Hality

TAMANHO_MAX = 20 * 1024 * 1024

app = FastAPI(
    title="Hality",
    description="Triagem de indicio de halitose a partir de foto da lingua. "
                "Nao e diagnostico.",
    version="0.1.0",
)

_modelo: Hality | None = None


def modelo() -> Hality:
    global _modelo
    if _modelo is None:
        _modelo = Hality()
    return _modelo


@app.get("/saude")
def saude() -> dict:
    try:
        m = modelo()
    except FileNotFoundError:
        raise HTTPException(503, "Modelos nao treinados. Rode hality.train_segmenter "
                                 "e hality.train_classifier.")
    return {
        "status": "ok",
        "auc_teste": round(m.auc_teste, 3),
        "iou_segmentador": round(m.iou_seg, 3),
        "limiar": round(m.limiar, 3),
        "faixa_abstencao": [round(m.abst_lo, 3), round(m.abst_hi, 3)],
        "aviso": "Ferramenta de triagem. Nao substitui avaliacao odontologica.",
    }


@app.post("/analisar")
async def analisar(foto: UploadFile = File(..., description="Foto da lingua")) -> JSONResponse:
    raw = await foto.read()
    if not raw:
        raise HTTPException(400, "Arquivo vazio.")
    if len(raw) > TAMANHO_MAX:
        raise HTTPException(413, f"Arquivo acima de {TAMANHO_MAX // 1024 // 1024} MB.")

    try:
        m = modelo()
    except FileNotFoundError:
        raise HTTPException(503, "Modelos nao treinados.")

    r = m.analisar(raw)
    corpo = r.dict()
    corpo["arquivo"] = foto.filename
    corpo["aviso"] = "Triagem, nao diagnostico. Procure um dentista para avaliacao."
    return JSONResponse(corpo, status_code=200)
