#!/usr/bin/env python3
"""
Benchmark complementar para contextualizar o tempo de treino em Rust.

Compara o MLP multivariado sem geração defasada do artigo contra duas implementações em
Python:
  1) Python puro, com laços explícitos e backprop manual;
  2) NumPy vetorizado, ainda sem bibliotecas de ML.

O caso em Python puro mede poucas épocas e estima o custo para 2.000 épocas
por extrapolação linear. O objetivo é medir custo computacional da rotina
forward+backward+Adam, não produzir novas métricas preditivas.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

PROJ = Path(__file__).resolve().parents[2]
DATASET = PROJ / "data" / "processed" / "dataset_final.csv"
OUT = PROJ / "data" / "outputs"
RUST_BENCH = OUT / "benchmark_rust.csv"

FEATURES = [
    "vazao_norm",
    "ena_bruta_norm",
    "mes_sin",
    "mes_cos",
    "vazao_lag1_norm",
]
TARGET = "geracao_norm"
ARQ = [len(FEATURES), 16, 16, 1]
EPOCAS_RUST = 2000
LR_PYTHON_PURO = 0.01
LR_NUMPY = 0.001


def carregar_dados() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]], list[float], list[list[float]]]:
    df = pd.read_csv(DATASET, parse_dates=["data"])
    treino = df[df["data"] <= "2022-12-31"].copy()
    teste = df[df["data"] >= "2023-01-01"].copy()

    x_treino = treino[FEATURES].to_numpy(dtype=np.float64)
    y_treino = treino[TARGET].to_numpy(dtype=np.float64)
    x_teste = teste[FEATURES].to_numpy(dtype=np.float64)

    return (
        x_treino,
        y_treino,
        x_teste,
        x_treino.tolist(),
        y_treino.tolist(),
        x_teste.tolist(),
    )


def inicializar_listas(semente: int = 17) -> tuple[list[list[list[float]]], list[list[float]]]:
    rng = random.Random(semente)
    pesos: list[list[list[float]]] = []
    vieses: list[list[float]] = []
    for fan_in, fan_out in zip(ARQ[:-1], ARQ[1:]):
        limite = math.sqrt(6.0 / (fan_in + fan_out))
        pesos.append([[rng.uniform(-limite, limite) for _ in range(fan_in)] for _ in range(fan_out)])
        vieses.append([0.0 for _ in range(fan_out)])
    return pesos, vieses


def zeros_like_pesos(pesos: list[list[list[float]]]) -> list[list[list[float]]]:
    return [[[0.0 for _ in linha] for linha in camada] for camada in pesos]


def zeros_like_vieses(vieses: list[list[float]]) -> list[list[float]]:
    return [[0.0 for _ in camada] for camada in vieses]


def forward_lista(
    pesos: list[list[list[float]]],
    vieses: list[list[float]],
    entrada: list[float],
) -> list[list[float]]:
    ativacoes = [entrada]
    for idx, (w_camada, b_camada) in enumerate(zip(pesos, vieses)):
        anterior = ativacoes[-1]
        saida = []
        ultima = idx == len(pesos) - 1
        for linha, b in zip(w_camada, b_camada):
            z = b
            for w, a in zip(linha, anterior):
                z += w * a
            saida.append(z if ultima else math.tanh(z))
        ativacoes.append(saida)
    return ativacoes


def epoca_python_puro(
    pesos: list[list[list[float]]],
    vieses: list[list[float]],
    x: list[list[float]],
    y: list[float],
    estado_adam: dict[str, object],
    lr: float = 0.01,
) -> float:
    grads_w = zeros_like_pesos(pesos)
    grads_b = zeros_like_vieses(vieses)
    perda = 0.0
    n = len(x)

    for entrada, alvo in zip(x, y):
        ativacoes = forward_lista(pesos, vieses, entrada)
        erro = ativacoes[-1][0] - alvo
        perda += erro * erro
        delta = [2.0 * erro]

        for i in range(len(pesos) - 1, -1, -1):
            anterior = ativacoes[i]
            for j in range(len(pesos[i])):
                grads_b[i][j] += delta[j]
                for k in range(len(pesos[i][j])):
                    grads_w[i][j][k] += delta[j] * anterior[k]

            if i > 0:
                novo_delta = []
                for k in range(len(ativacoes[i])):
                    soma = 0.0
                    for j in range(len(pesos[i])):
                        soma += pesos[i][j][k] * delta[j]
                    a = ativacoes[i][k]
                    novo_delta.append(soma * (1.0 - a * a))
                delta = novo_delta

    inv_n = 1.0 / n
    for i in range(len(grads_w)):
        for j in range(len(grads_w[i])):
            grads_b[i][j] *= inv_n
            for k in range(len(grads_w[i][j])):
                grads_w[i][j][k] *= inv_n

    adam_update_listas(pesos, vieses, grads_w, grads_b, estado_adam, lr)
    return perda * inv_n


def adam_update_listas(
    pesos: list[list[list[float]]],
    vieses: list[list[float]],
    grads_w: list[list[list[float]]],
    grads_b: list[list[float]],
    estado: dict[str, object],
    lr: float,
) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    estado["t"] = int(estado["t"]) + 1
    t = int(estado["t"])
    mw = estado["mw"]
    vw = estado["vw"]
    mb = estado["mb"]
    vb = estado["vb"]
    assert isinstance(mw, list) and isinstance(vw, list)
    assert isinstance(mb, list) and isinstance(vb, list)
    corr1 = 1.0 - beta1**t
    corr2 = 1.0 - beta2**t

    for i in range(len(pesos)):
        for j in range(len(pesos[i])):
            g_b = grads_b[i][j]
            mb[i][j] = beta1 * mb[i][j] + (1.0 - beta1) * g_b
            vb[i][j] = beta2 * vb[i][j] + (1.0 - beta2) * g_b * g_b
            m_hat_b = mb[i][j] / corr1
            v_hat_b = vb[i][j] / corr2
            vieses[i][j] -= lr * m_hat_b / (math.sqrt(v_hat_b) + eps)

            for k in range(len(pesos[i][j])):
                g = grads_w[i][j][k]
                mw[i][j][k] = beta1 * mw[i][j][k] + (1.0 - beta1) * g
                vw[i][j][k] = beta2 * vw[i][j][k] + (1.0 - beta2) * g * g
                m_hat = mw[i][j][k] / corr1
                v_hat = vw[i][j][k] / corr2
                pesos[i][j][k] -= lr * m_hat / (math.sqrt(v_hat) + eps)


def benchmark_python_puro(
    x_train: list[list[float]],
    y_train: list[float],
    x_test: list[list[float]],
    epocas: int,
) -> dict[str, object]:
    pesos, vieses = inicializar_listas()
    estado = {
        "t": 0,
        "mw": zeros_like_pesos(pesos),
        "vw": zeros_like_pesos(pesos),
        "mb": zeros_like_vieses(vieses),
        "vb": zeros_like_vieses(vieses),
    }

    epoca_python_puro(pesos, vieses, x_train, y_train, estado, lr=LR_PYTHON_PURO)
    inicio = time.perf_counter()
    perda = 0.0
    for _ in range(epocas):
        perda = epoca_python_puro(pesos, vieses, x_train, y_train, estado, lr=LR_PYTHON_PURO)
    treino_s = time.perf_counter() - inicio

    inicio = time.perf_counter()
    soma = 0.0
    for entrada in x_test:
        soma += forward_lista(pesos, vieses, entrada)[-1][0]
    infer_ms = (time.perf_counter() - inicio) * 1000.0
    if not math.isfinite(soma):
        raise RuntimeError("inferência Python puro produziu valor não finito")

    por_epoca_ms = 1000.0 * treino_s / epocas
    return {
        "implementacao": "python_puro_loops",
        "modelo": "MLP multivariado",
        "features": "+".join(FEATURES),
        "n_treino": len(x_train),
        "n_teste": len(x_test),
        "epocas_medidas": epocas,
        "tempo_treino_medido_s": treino_s,
        "tempo_por_epoca_ms": por_epoca_ms,
        "tempo_2000_epocas_estimado_s": por_epoca_ms * EPOCAS_RUST / 1000.0,
        "tempo_inferencia_teste_ms": infer_ms,
        "observacao": (
            f"medido por {epocas} épocas; 2000 épocas estimadas; "
            f"Adam lr={LR_PYTHON_PURO}; loss_final={perda:.6f}"
        ),
    }


def inicializar_numpy(semente: int = 17) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(semente)
    pesos = []
    vieses = []
    for fan_in, fan_out in zip(ARQ[:-1], ARQ[1:]):
        limite = math.sqrt(6.0 / (fan_in + fan_out))
        pesos.append(rng.uniform(-limite, limite, size=(fan_out, fan_in)))
        vieses.append(np.zeros((fan_out,), dtype=np.float64))
    return pesos, vieses


def adam_update_numpy(
    pesos: list[np.ndarray],
    vieses: list[np.ndarray],
    grads_w: list[np.ndarray],
    grads_b: list[np.ndarray],
    estado: dict[str, object],
    lr: float = 0.01,
) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    estado["t"] = int(estado["t"]) + 1
    t = int(estado["t"])
    mw = estado["mw"]
    vw = estado["vw"]
    mb = estado["mb"]
    vb = estado["vb"]
    assert isinstance(mw, list) and isinstance(vw, list)
    assert isinstance(mb, list) and isinstance(vb, list)
    corr1 = 1.0 - beta1**t
    corr2 = 1.0 - beta2**t
    for i in range(len(pesos)):
        mw[i] = beta1 * mw[i] + (1.0 - beta1) * grads_w[i]
        vw[i] = beta2 * vw[i] + (1.0 - beta2) * grads_w[i] * grads_w[i]
        mb[i] = beta1 * mb[i] + (1.0 - beta1) * grads_b[i]
        vb[i] = beta2 * vb[i] + (1.0 - beta2) * grads_b[i] * grads_b[i]
        pesos[i] -= lr * (mw[i] / corr1) / (np.sqrt(vw[i] / corr2) + eps)
        vieses[i] -= lr * (mb[i] / corr1) / (np.sqrt(vb[i] / corr2) + eps)
    estado["mw"] = mw
    estado["vw"] = vw
    estado["mb"] = mb
    estado["vb"] = vb


def epoca_numpy(
    pesos: list[np.ndarray],
    vieses: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    estado: dict[str, object],
    lr: float = 0.01,
) -> float:
    z1 = x @ pesos[0].T + vieses[0]
    a1 = np.tanh(z1)
    z2 = a1 @ pesos[1].T + vieses[1]
    a2 = np.tanh(z2)
    pred = a2 @ pesos[2].T + vieses[2]
    erro = pred[:, 0] - y
    perda = float(np.mean(erro * erro))
    if not math.isfinite(perda):
        raise RuntimeError("treino NumPy produziu loss não finita")

    n = x.shape[0]
    dz3 = (2.0 / n) * erro[:, None]
    dw3 = dz3.T @ a2
    db3 = dz3.sum(axis=0)
    dz2 = (dz3 @ pesos[2]) * (1.0 - a2 * a2)
    dw2 = dz2.T @ a1
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ pesos[1]) * (1.0 - a1 * a1)
    dw1 = dz1.T @ x
    db1 = dz1.sum(axis=0)

    adam_update_numpy(pesos, vieses, [dw1, dw2, dw3], [db1, db2, db3], estado, lr)
    return perda


def infer_numpy(pesos: list[np.ndarray], vieses: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    a1 = np.tanh(x @ pesos[0].T + vieses[0])
    a2 = np.tanh(a1 @ pesos[1].T + vieses[1])
    return a2 @ pesos[2].T + vieses[2]


def benchmark_numpy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, epocas: int) -> dict[str, object]:
    pesos, vieses = inicializar_numpy()
    estado = {
        "t": 0,
        "mw": [np.zeros_like(w) for w in pesos],
        "vw": [np.zeros_like(w) for w in pesos],
        "mb": [np.zeros_like(b) for b in vieses],
        "vb": [np.zeros_like(b) for b in vieses],
    }

    epoca_numpy(pesos, vieses, x_train, y_train, estado, lr=LR_NUMPY)
    inicio = time.perf_counter()
    perda = 0.0
    for _ in range(epocas):
        perda = epoca_numpy(pesos, vieses, x_train, y_train, estado, lr=LR_NUMPY)
    treino_s = time.perf_counter() - inicio

    inicio = time.perf_counter()
    pred = infer_numpy(pesos, vieses, x_test)
    infer_ms = (time.perf_counter() - inicio) * 1000.0
    if not np.isfinite(pred).all():
        raise RuntimeError("inferência NumPy produziu valor não finito")

    por_epoca_ms = 1000.0 * treino_s / epocas
    return {
        "implementacao": "python_numpy_vetorizado",
        "modelo": "MLP multivariado",
        "features": "+".join(FEATURES),
        "n_treino": x_train.shape[0],
        "n_teste": x_test.shape[0],
        "epocas_medidas": epocas,
        "tempo_treino_medido_s": treino_s,
        "tempo_por_epoca_ms": por_epoca_ms,
        "tempo_2000_epocas_estimado_s": por_epoca_ms * EPOCAS_RUST / 1000.0,
        "tempo_inferencia_teste_ms": infer_ms,
        "observacao": f"NumPy vetorizado; Adam lr={LR_NUMPY}; loss_final={perda:.6f}",
    }


def linhas_rust_comparaveis() -> list[dict[str, object]]:
    if not RUST_BENCH.exists():
        return []
    linhas = []
    with RUST_BENCH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["modelo"] != "mlp_multivariado":
                continue
            treino_s = float(row["tempo_treino_s"])
            infer_ms = float(row["tempo_inferencia_teste_ms"])
            por_epoca_ms = 1000.0 * treino_s / EPOCAS_RUST
            linhas.append(
                {
                    "implementacao": "rust_release",
                    "modelo": "MLP multivariado",
                    "features": "+".join(FEATURES),
                    "n_treino": int(row["n_treino"]),
                    "n_teste": int(row["n_teste"]),
                    "epocas_medidas": EPOCAS_RUST,
                    "tempo_treino_medido_s": treino_s,
                    "tempo_por_epoca_ms": por_epoca_ms,
                    "tempo_2000_epocas_estimado_s": treino_s,
                    "tempo_inferencia_teste_ms": infer_ms,
                    "observacao": "binário Rust release; backprop manual; Adam; 2000 épocas medidas",
                }
            )
    return linhas


def escrever_csv(caminho: Path, linhas: Iterable[dict[str, object]]) -> None:
    linhas = list(linhas)
    if not linhas:
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "implementacao",
        "modelo",
        "features",
        "n_treino",
        "n_teste",
        "epocas_medidas",
        "tempo_treino_medido_s",
        "tempo_por_epoca_ms",
        "tempo_2000_epocas_estimado_s",
        "tempo_inferencia_teste_ms",
        "observacao",
    ]
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for linha in linhas:
            writer.writerow(linha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-epochs", type=int, default=10)
    parser.add_argument("--numpy-epochs", type=int, default=2000)
    args = parser.parse_args()

    x_np, y_np, xt_np, x_list, y_list, xt_list = carregar_dados()
    python_rows = [
        benchmark_python_puro(x_list, y_list, xt_list, args.pure_epochs),
        benchmark_numpy(x_np, y_np, xt_np, args.numpy_epochs),
    ]
    escrever_csv(OUT / "benchmark_python.csv", python_rows)

    comparativo = linhas_rust_comparaveis() + python_rows
    escrever_csv(OUT / "benchmark_comparativo.csv", comparativo)

    print("Benchmark Python salvo em data/outputs/benchmark_python.csv")
    print("Comparativo salvo em data/outputs/benchmark_comparativo.csv")
    for row in comparativo:
        print(
            f"{row['implementacao']:<24} "
            f"treino medido={float(row['tempo_treino_medido_s']):8.3f}s | "
            f"por época={float(row['tempo_por_epoca_ms']):8.3f}ms | "
            f"2000 épocas≈{float(row['tempo_2000_epocas_estimado_s']):8.3f}s"
        )


if __name__ == "__main__":
    main()
