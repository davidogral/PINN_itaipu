#!/usr/bin/env python3
"""
Lê os CSVs brutos de data/raw/ e produz três tabelas limpas, diárias e
com data padronizada (YYYY-MM-DD), em data/processed/:

  1. geracao_diaria.csv   <- agrega a geração HORÁRIA para média diária (MWmed)
  2. ena_parana.csv       <- ENA da bacia PARANA (bruta e armazenável, MWmed)
  3. vazao_porto_sao_jose.csv <- vazão diária do posto PORTO SAO JOSE (m³/s)

Decisões aplicadas (ver docs/decisoes_tecnicas.md §6 e §8):
  - geração diária = MÉDIA das horas do dia (MWmed) -> casa com a unidade da ENA.
    (energia diária em MWh = geracao_mwmed * 24)
  - ENA: variável principal = ena_bruta_bacia_mwmed (bacia "PARANA", match exato).
  - vazão: complementar à EDA = posto "PORTO SAO JOSE" no "RIO PARANÁ".
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJ_ROOT / "data" / "raw"
PROC = PROJ_ROOT / "data" / "processed"

# Mínimo de horas para considerar um dia "completo" (evita viés de dia parcial).
MIN_HORAS_DIA = 20


def limpar_geracao() -> pd.DataFrame:
    print("[1/3] Geração de Itaipu (horária -> diária)")
    df = pd.read_csv(RAW / "GERACAO_ITAIPU.csv", sep=";",
                     usecols=["din_instante", "val_itaipu_total"])
    df["din_instante"] = pd.to_datetime(df["din_instante"], errors="coerce")
    df["val_itaipu_total"] = pd.to_numeric(df["val_itaipu_total"], errors="coerce")
    df = df.dropna(subset=["din_instante", "val_itaipu_total"])

    df["data"] = df["din_instante"].dt.normalize()
    diaria = (df.groupby("data")["val_itaipu_total"]
                .agg(geracao_mwmed="mean", n_horas="count")
                .reset_index())

    antes = len(diaria)
    diaria = diaria[diaria["n_horas"] >= MIN_HORAS_DIA].copy()
    print(f"   dias: {antes}  ->  {len(diaria)} (>= {MIN_HORAS_DIA} h/dia)")
    print(f"   período: {diaria['data'].min().date()} -> {diaria['data'].max().date()}")
    print(f"   geração diária (MWmed): min={diaria.geracao_mwmed.min():.0f} "
          f"max={diaria.geracao_mwmed.max():.0f}")
    diaria.to_csv(PROC / "geracao_diaria.csv", index=False)
    return diaria


def limpar_ena() -> pd.DataFrame:
    print("[2/3] ENA — bacia PARANA")
    arquivos = sorted(glob.glob(str(RAW / "ENA_DIARIO_BACIAS_*.csv")))
    partes = []
    for a in arquivos:
        d = pd.read_csv(a, sep=";",
                        usecols=["nom_bacia", "ena_data",
                                 "ena_bruta_bacia_mwmed",
                                 "ena_armazenavel_bacia_mwmed"])
        partes.append(d[d["nom_bacia"] == "PARANA"])
    ena = pd.concat(partes, ignore_index=True)
    ena["data"] = pd.to_datetime(ena["ena_data"], errors="coerce").dt.normalize()
    ena["ena_bruta"] = pd.to_numeric(ena["ena_bruta_bacia_mwmed"], errors="coerce")
    ena["ena_armazenavel"] = pd.to_numeric(ena["ena_armazenavel_bacia_mwmed"],
                                           errors="coerce")
    ena = (ena.dropna(subset=["data", "ena_bruta"])
              .loc[:, ["data", "ena_bruta", "ena_armazenavel"]]
              .sort_values("data")
              .reset_index(drop=True))
    print(f"   dias: {len(ena)}  | período: {ena.data.min().date()} -> "
          f"{ena.data.max().date()}")
    print(f"   ENA bruta (MWmed): min={ena.ena_bruta.min():.0f} "
          f"max={ena.ena_bruta.max():.0f}")
    ena.to_csv(PROC / "ena_parana.csv", index=False)
    return ena


def limpar_vazao() -> pd.DataFrame:
    print("[3/3] Vazão — posto PORTO SAO JOSE (complementar)")
    arquivos = sorted(glob.glob(str(RAW / "GRANDEZAS_FLUVIOMETRICAS_*.csv")))
    partes = []
    for a in arquivos:
        d = pd.read_csv(a, sep=";",
                        usecols=["nom_postofluviometrico", "nom_rio",
                                 "din_medicao", "val_vazaomedia"])
        partes.append(d[d["nom_postofluviometrico"] == "PORTO SAO JOSE"])
    vaz = pd.concat(partes, ignore_index=True)
    vaz["data"] = pd.to_datetime(vaz["din_medicao"], errors="coerce").dt.normalize()
    vaz["vazao"] = pd.to_numeric(vaz["val_vazaomedia"], errors="coerce")
    vaz = vaz.dropna(subset=["data", "vazao"])
    # O posto PORTO SAO JOSE tem DOIS medidores (id 64575000 e 64575001) com
    # leituras distintas no mesmo dia. Agregamos por data (média) para ter
    # uma única série diária e evitar explosão no join.
    vaz = (vaz.groupby("data", as_index=False)["vazao"].mean()
              .sort_values("data")
              .reset_index(drop=True))
    if vaz.empty:
        print("   ⚠️ nenhuma linha para PORTO SAO JOSE (conferir filtro)")
    else:
        print(f"   dias: {len(vaz)}  | período: {vaz.data.min().date()} -> "
              f"{vaz.data.max().date()}")
        print(f"   vazão (m³/s): min={vaz.vazao.min():.0f} max={vaz.vazao.max():.0f}")
    vaz.to_csv(PROC / "vazao_porto_sao_jose.csv", index=False)
    return vaz


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Fase 2 — limpeza e padronização")
    print("=" * 60)
    limpar_geracao()
    limpar_ena()
    limpar_vazao()
    print("\n✓ Tabelas limpas salvas em data/processed/")
    print("  Próximo passo: cruzar_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
