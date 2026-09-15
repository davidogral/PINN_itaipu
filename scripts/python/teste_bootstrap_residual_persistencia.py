#!/usr/bin/env python3
"""Bootstrap pareado em blocos para comparar persistencia e modelos residuais.

O teste reamostra blocos consecutivos do periodo de teste, preservando parte da
autocorrelacao diaria, e estima o intervalo de confianca da reducao de RMSE:

    delta = RMSE(persistencia) - RMSE(modelo)

Valores positivos favorecem o modelo residual.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"
N_BOOT = 20_000
SEED = 20_260_811
BLOCKS = (7, 14, 30)


def carregar_predicoes_validacao_temporal() -> tuple[list[float], list[float], list[float]]:
    caminho = OUT / "predicoes_validacao_temporal.csv"
    real: list[float] = []
    persistencia: list[float] = []
    residual: list[float] = []
    with caminho.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["periodo"] != "teste":
                continue
            real.append(float(row["geracao_real_mwmed"]))
            persistencia.append(float(row["persistencia_dia_anterior_mwmed"]))
            residual.append(float(row["persistencia_residual_mlp_mwmed"]))
    return real, persistencia, residual


def carregar_predicoes_hidro_longo() -> tuple[list[float], list[float], list[float]]:
    caminho = OUT / "predicoes_exploracao_temporal.csv"
    real: list[float] = []
    persistencia: list[float] = []
    residual: list[float] = []
    with caminho.open(newline="") as f:
        for row in csv.DictReader(f):
            real.append(float(row["geracao_real_mwmed"]))
            persistencia.append(float(row["persistencia_mwmed"]))
            residual.append(float(row["mlp_residual_hidro_longo_mwmed"]))
    return real, persistencia, residual


def rmse(real: list[float], pred: list[float], idx: list[int] | None = None) -> float:
    if idx is None:
        idx = list(range(len(real)))
    return math.sqrt(sum((real[i] - pred[i]) ** 2 for i in idx) / len(idx))


def mae(real: list[float], pred: list[float]) -> float:
    return sum(abs(r - p) for r, p in zip(real, pred)) / len(real)


def bootstrap_delta_rmse(
    real: list[float],
    persistencia: list[float],
    modelo: list[float],
    bloco: int,
) -> tuple[float, float, float, float]:
    rng = random.Random(SEED + bloco)
    n = len(real)
    starts = list(range(0, n - bloco + 1))
    deltas: list[float] = []

    for _ in range(N_BOOT):
        idx: list[int] = []
        while len(idx) < n:
            ini = rng.choice(starts)
            idx.extend(range(ini, ini + bloco))
        idx = idx[:n]
        deltas.append(rmse(real, persistencia, idx) - rmse(real, modelo, idx))

    deltas.sort()
    low = deltas[int(0.025 * N_BOOT)]
    med = deltas[int(0.500 * N_BOOT)]
    high = deltas[int(0.975 * N_BOOT)]
    p_le_zero = sum(1 for d in deltas if d <= 0.0) / N_BOOT
    return low, med, high, p_le_zero


def linha_resultado(
    cenario: str,
    real: list[float],
    persistencia: list[float],
    modelo: list[float],
    bloco: int,
) -> dict[str, str]:
    rmse_p = rmse(real, persistencia)
    rmse_m = rmse(real, modelo)
    low, med, high, p_le_zero = bootstrap_delta_rmse(real, persistencia, modelo, bloco)
    return {
        "cenario": cenario,
        "n": str(len(real)),
        "bloco_dias": str(bloco),
        "reamostragens": str(N_BOOT),
        "rmse_persistencia_mwmed": f"{rmse_p:.2f}",
        "rmse_modelo_mwmed": f"{rmse_m:.2f}",
        "delta_rmse_mwmed": f"{rmse_p - rmse_m:.2f}",
        "mae_persistencia_mwmed": f"{mae(real, persistencia):.2f}",
        "mae_modelo_mwmed": f"{mae(real, modelo):.2f}",
        "delta_mae_mwmed": f"{mae(real, persistencia) - mae(real, modelo):.2f}",
        "ic95_delta_rmse_inf_mwmed": f"{low:.2f}",
        "mediana_delta_rmse_mwmed": f"{med:.2f}",
        "ic95_delta_rmse_sup_mwmed": f"{high:.2f}",
        "p_delta_menor_igual_zero": f"{p_le_zero:.4f}",
    }


def main() -> None:
    cenarios = [
        ("residual_principal_5_variaveis", carregar_predicoes_validacao_temporal()),
        ("residual_hidrologico_lags_30d", carregar_predicoes_hidro_longo()),
    ]
    linhas = []
    for cenario, series in cenarios:
        real, persistencia, modelo = series
        for bloco in BLOCKS:
            linhas.append(linha_resultado(cenario, real, persistencia, modelo, bloco))

    caminho_saida = OUT / "bootstrap_residual_persistencia.csv"
    with caminho_saida.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)

    print(f"[bootstrap] artefato: {caminho_saida}")
    for row in linhas:
        if row["bloco_dias"] == "14":
            print(
                "[bootstrap] {cenario}: delta RMSE={delta_rmse_mwmed} "
                "IC95=[{ic95_delta_rmse_inf_mwmed}, {ic95_delta_rmse_sup_mwmed}] "
                "p(delta<=0)={p_delta_menor_igual_zero}".format(**row)
            )


if __name__ == "__main__":
    main()
