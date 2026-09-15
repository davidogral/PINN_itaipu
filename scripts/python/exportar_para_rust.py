#!/usr/bin/env python3
"""
Gera o ÚNICO arquivo de contato entre Python e Rust:

    data/processed/dataset_final.csv   (separador vírgula, p/ o crate `csv`)

Conteúdo (núcleo do modelo, decisão §8):
    data, vazao, ena_bruta, mes_sin, mes_cos, vazao_lag1, geracao_lag1,
    geracao, *_norm

  - vazao    = vazão diária do posto PORTO SAO JOSE (m³/s)  [entrada do modelo]
               proxy de disponibilidade hídrica a montante, não vazão turbinada
  - ena_bruta = Energia Natural Afluente da bacia Paraná (MWmed)
  - mes_sin/cos = sazonalidade anual codificada de forma cíclica
  - *_lag1   = valores observados no dia anterior
  - geracao  = geração diária média de Itaipu (MWmed)       [alvo do modelo]
  - *_norm   = versões normalizadas em [0, 1] com min/max do treino

Período efetivo: 2015–2024 (a vazão fluviométrica só existe a partir de 2015).
Normalização ajustada apenas em 2015–2022 para evitar vazamento temporal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJ_ROOT / "data" / "processed"


DATA_LIMITE_TREINO = pd.Timestamp("2022-12-31")


def aplicar_minmax(final: pd.DataFrame, coluna: str, mascara_treino: pd.Series) -> dict[str, float | str]:
    lo = float(final.loc[mascara_treino, coluna].min())
    hi = float(final.loc[mascara_treino, coluna].max())
    if hi > lo:
        final[f"{coluna}_norm"] = (final[coluna] - lo) / (hi - lo)
    else:
        final[f"{coluna}_norm"] = 0.0
    return {"min": lo, "max": hi}


def main() -> int:
    print("=" * 60)
    print("  Fase 2 — exportação para o Rust")
    print("=" * 60)

    cruz = pd.read_csv(PROC / "dataset_cruzado.csv", parse_dates=["data"])

    final = cruz.rename(columns={"geracao_mwmed": "geracao"})
    # Entrada do modelo = proxy de vazão afluente. dropna remove o período < 2015.
    final = final[["data", "vazao", "ena_bruta", "geracao"]].dropna().reset_index(drop=True)
    final["mes"] = final["data"].dt.month
    final["mes_sin"] = (2.0 * math.pi * final["mes"] / 12.0).map(math.sin)
    final["mes_cos"] = (2.0 * math.pi * final["mes"] / 12.0).map(math.cos)
    final["vazao_lag1"] = final["vazao"].shift(1)
    final["geracao_lag1"] = final["geracao"].shift(1)
    final = final.dropna().reset_index(drop=True)

    mascara_treino = final["data"] <= DATA_LIMITE_TREINO
    params = {}
    for coluna, unidade in [
        ("vazao", "m3/s"),
        ("ena_bruta", "MWmed"),
        ("vazao_lag1", "m3/s"),
        ("geracao_lag1", "MWmed"),
        ("geracao", "MWmed"),
    ]:
        params[coluna] = aplicar_minmax(final, coluna, mascara_treino)
        params[coluna]["unidade"] = unidade

    final["data"] = final["data"].dt.strftime("%Y-%m-%d")
    final = final[
        [
            "data",
            "vazao",
            "ena_bruta",
            "mes_sin",
            "mes_cos",
            "vazao_lag1",
            "geracao_lag1",
            "geracao",
            "vazao_norm",
            "ena_bruta_norm",
            "vazao_lag1_norm",
            "geracao_lag1_norm",
            "geracao_norm",
        ]
    ]
    final.to_csv(PROC / "dataset_final.csv", index=False)

    params.update({
        "metodo": "min-max [0,1]",
        "ajuste_normalizacao": "somente treino 2015-2022",
        "n_amostras": int(len(final)),
        "n_treino_2015_2022": int((pd.to_datetime(final["data"]) <= DATA_LIMITE_TREINO).sum()),
        "n_teste_2023_2024": int((pd.to_datetime(final["data"]) >= pd.Timestamp("2023-01-01")).sum()),
    })
    (PROC / "normalizacao.json").write_text(json.dumps(params, indent=2,
                                                       ensure_ascii=False))

    print(f"amostras finais: {len(final)}")
    print(f"período: {final.data.iloc[0]} -> {final.data.iloc[-1]}")
    print(f"normalização ajustada em: 2015-2022 ({params['n_treino_2015_2022']} amostras)")
    print(f"teste temporal: 2023-2024 ({params['n_teste_2023_2024']} amostras)")
    print(f"vazao    [{params['vazao']['min']:.0f}, {params['vazao']['max']:.0f}] m³/s")
    print(f"geracao  [{params['geracao']['min']:.0f}, {params['geracao']['max']:.0f}] MWmed")
    print("\n✓ data/processed/dataset_final.csv")
    print("✓ data/processed/normalizacao.json")
    print("  Fase 2 concluída — pronto para o Rust (Fase 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
