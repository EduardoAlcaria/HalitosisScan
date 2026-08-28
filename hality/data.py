"""Reconciliacao: constroi a tabela mestra e a divisao por paciente.

As pastas soltas do repositorio (`data/Imagens lingua/`, `data/Classificacao/`,
`data/classificacao_rotulados/`, os tres `dataset*` do drive) sao entrada bruta e nao
sao consumidas diretamente pelo treino. Tudo passa por aqui.

Divisao por PACIENTE, nunca por foto: um paciente com mais de uma imagem, ou uma foto
que gera varios recortes, precisa ficar inteiro de um lado so. Caso contrario o teste
mede memorizacao.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGADO = os.path.join(ROOT, "Hality-Project-main")
CSV = os.path.join(LEGADO, "data", "TabelaHality_Clean.csv")
FOTOS = os.path.join(LEGADO, "data", "Classificacao")
MASCARAS = os.path.join(LEGADO, "recortadas", "recortadas")

MIN_AREA_MASCARA = 0.15   # abaixo disso a segmentacao falhou por sub-segmentacao


@dataclass
class Amostra:
    pid: str
    foto: str
    mascara: str
    nota: int          # 1, 2 ou 3, como veio da clinica
    alvo: int          # 1 se nota == 3, senao 0
    area: float


def tabela_mestra(exigir_area: bool = True) -> list[Amostra]:
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    coluna_nota = df.columns[-1]
    notas = dict(zip(df["ID_ANAMNESE"].astype(str).str.lower(), df[coluna_nota]))

    fotos = {os.path.splitext(os.path.basename(f))[0].lower(): f
             for f in glob.glob(os.path.join(FOTOS, "*"))}

    out: list[Amostra] = []
    for m in sorted(glob.glob(os.path.join(MASCARAS, "*_mask.png"))):
        pid = os.path.basename(m).replace("_mask.png", "").lower()
        if pid not in fotos or pid not in notas:
            continue
        try:
            area = (np.asarray(Image.open(m).convert("L")) > 127).mean()
        except Exception:
            continue
        if exigir_area and area < MIN_AREA_MASCARA:
            continue          # sub-segmentada: rotulo de mascara nao confiavel
        nota = int(notas[pid])
        out.append(Amostra(pid, fotos[pid], m, nota, int(nota == 3), float(area)))
    return out


def dividir(amostras: list[Amostra], seed: int = 42) -> dict[str, list[Amostra]]:
    """Estratificada pelo alvo binario, agrupada por paciente, 70/15/15.

    Hoje cada paciente tem uma foto, entao agrupar nao muda nada -- mas a regra fica
    escrita para quando chegar um segundo lote com varias fotos por pessoa.
    """
    rng = np.random.default_rng(seed)
    part: dict[str, list[Amostra]] = {"treino": [], "val": [], "teste": []}
    for alvo in (0, 1):
        g = [a for a in amostras if a.alvo == alvo]
        idx = rng.permutation(len(g))
        n_tr, n_va = int(0.70 * len(g)), int(0.15 * len(g))
        for k, i in enumerate(idx):
            nome = "treino" if k < n_tr else ("val" if k < n_tr + n_va else "teste")
            part[nome].append(g[i])
    return part


def carregar(a: Amostra, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (rgb uint8 [size,size,3], mask bool [size,size])."""
    im = Image.open(a.foto).convert("RGB")
    mk = Image.open(a.mascara).convert("L")
    if mk.size != im.size:
        mk = mk.resize(im.size, Image.NEAREST)
    rgb = np.asarray(im.resize((size, size), Image.BICUBIC))
    mask = np.asarray(mk.resize((size, size), Image.NEAREST)) > 127
    return rgb, mask


def demo() -> None:
    am = tabela_mestra()
    print(f"amostras com mascara utilizavel: {len(am)}")
    todas = tabela_mestra(exigir_area=False)
    print(f"descartadas por sub-segmentacao: {len(todas) - len(am)}")

    p = dividir(am)
    n = {k: len(v) for k, v in p.items()}
    print("particoes:", n, "total", sum(n.values()))
    assert sum(n.values()) == len(am)

    # nenhum paciente em duas particoes
    ids = [{a.pid for a in v} for v in p.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])

    # estratificacao preservada dentro de uma folga razoavel
    base = np.mean([a.alvo for a in am])
    for k, v in p.items():
        prop = np.mean([a.alvo for a in v])
        print(f"  {k}: n={len(v)} prevalencia={prop:.3f}")
        assert abs(prop - base) < 0.08, (k, prop, base)

    rgb, mask = carregar(am[0], 128)
    assert rgb.shape == (128, 128, 3) and mask.shape == (128, 128)
    print("ok")


if __name__ == "__main__":
    demo()
