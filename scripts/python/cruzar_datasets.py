#!/usr/bin/env python3
"""
Cruza as tabelas limpas de data/processed/ em uma única tabela diária.

Núcleo do modelo: [data, ena_bruta, geracao_mwmed]
  - join INNER entre geração diária e ENA da bacia PARANA (overlap 2000–2024).
  - join LEFT da vazão (PORTO SAO JOSE) como coluna complementar
    (fica vazia antes de 2015 — usada só na EDA).

Também imprime as correlações com a geração para validar a decisão de usar
a ENA BRUTA (deve correlacionar mais que a armazenável e que a vazão).

Saída: data/processed/dataset_cruzado.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJ_ROOT / "data" / "processed"


def main() -> int:
    print("=" * 60)
    print("  Fase 2 — cruzamento dos datasets (join por data)")
    print("=" * 60)

    ger = pd.read_csv(PROC / "geracao_diaria.csv", parse_dates=["data"])
    ena = pd.read_csv(PROC / "ena_parana.csv", parse_dates=["data"])
    vaz = pd.read_csv(PROC / "vazao_porto_sao_jose.csv", parse_dates=["data"])

    # Núcleo: geração × ENA (inner) -> período comum
    core = pd.merge(ger[["data", "geracao_mwmed"]], ena, on="data", how="inner")
    # Complementar: vazão (left) -> NaN antes de 2015
    cruz = pd.merge(core, vaz, on="data", how="left").sort_values("data")
    cruz = cruz.reset_index(drop=True)

    print(f"\nlinhas no cruzamento (núcleo ENA×geração): {len(cruz)}")
    print(f"período: {cruz.data.min().date()} -> {cruz.data.max().date()}")
    n_vaz = cruz["vazao"].notna().sum()
    print(f"dias com vazão disponível (>=2015): {n_vaz}")

    # Validação das correlações com a geração 
    print("\nCorrelação (Pearson) com geracao_mwmed:")
    for col in ["ena_bruta", "ena_armazenavel", "vazao"]:
        sub = cruz[["geracao_mwmed", col]].dropna()
        r = sub["geracao_mwmed"].corr(sub[col]) if len(sub) > 2 else float("nan")
        print(f"   {col:18s} r = {r:+.3f}   (n={len(sub)})")

    cruz.to_csv(PROC / "dataset_cruzado.csv", index=False)
    print("\n✓ data/processed/dataset_cruzado.csv salvo")
    print("  Próximo passo: exportar_para_rust.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
