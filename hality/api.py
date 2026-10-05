from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from .pipeline import Hality

TAMANHO_MAX = 20 * 1024 * 1024
AVISO = "Triagem, nao diagnostico. Procure um dentista para avaliacao."

_modelo: Hality | None = None


def _aquecer(m: Hality) -> None:
    buf = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(buf, "PNG")
    m.analisar(buf.getvalue())
    if m.gate is not None:
        m.tongue_gate(np.full((256, 256, 3), 128, np.uint8))


@asynccontextmanager
async def ciclo(app: FastAPI):
    global _modelo
    _modelo = Hality()
    await run_in_threadpool(_aquecer, _modelo)
    yield


app = FastAPI(
    title="Hality",
    description="Triagem de indicio de halitose a partir de foto da lingua. Nao e diagnostico.",
    version="0.2.0",
    lifespan=ciclo,
)


@app.get("/saude")
def saude() -> dict:
    m = _modelo
    if m is None:
        raise HTTPException(503, "Modelos ainda carregando.")
    return {
        "status": "ok",
        "segmentador": m.seg.nome,
        "gate": m.gate is not None,
        "auc_teste": round(m.auc_teste, 3),
        "iou_segmentador": round(m.iou_seg, 3),
        "limiar": round(m.limiar, 3),
        "faixa_abstencao": [round(m.abst_lo, 3), round(m.abst_hi, 3)],
        "aviso": AVISO,
    }


@app.post("/analisar")
async def analisar(foto: UploadFile = File(..., description="Foto da lingua")) -> JSONResponse:
    raw = await foto.read()
    if not raw:
        raise HTTPException(400, "Arquivo vazio.")
    if len(raw) > TAMANHO_MAX:
        raise HTTPException(413, f"Arquivo acima de {TAMANHO_MAX // 1024 // 1024} MB.")
    if _modelo is None:
        raise HTTPException(503, "Modelos ainda carregando.")

    t0 = time.perf_counter()
    r = await run_in_threadpool(_modelo.analisar, raw)
    corpo = r.dict()
    corpo["arquivo"] = foto.filename
    corpo["tempo_ms"] = round((time.perf_counter() - t0) * 1000)
    corpo["aviso"] = AVISO
    return JSONResponse(corpo, status_code=200)
