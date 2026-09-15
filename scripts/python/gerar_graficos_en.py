#!/usr/bin/env python3
"""
English version of `gerar_graficos.py`.

Reads the results produced by the Rust physics-guided network
(data/outputs/*.csv) plus the processed dataset, and renders the paper figures
with English labels into artigo/figuras_en/ (used by artigo/main_en.tex).

Figures:
  fig1_conceitual.png           conceptual view: flow proxy -> saturating curve
  fig2_arquitetura.png          MLP vs. physics-guided network with backprop
  fig3_ajuste_pinn.png          train/test scatter + curve trained up to 2022
  fig4_convergencia_loss.png    convergence of the 3 loss terms
  fig5_eficiencia.png           relative productivity + saturation knee
  fig6_retrospectiva.png        temporal test: monthly residuals + absolute error
  fig7_ablacao.png              3 curves of lambda_physics (physics residual effect)
  fig8_pinn_vs_interpolacao.png ML metrics on the temporal test, with baselines
  fig9_vazao_turbinada_eq.png   equivalent turbined discharge diagnostic

Usage: python scripts/python/gerar_graficos_en.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pinn_itaipu_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pinn_itaipu_cache"))

import matplotlib

matplotlib.use("Agg")  # headless backend (writes PNGs)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
PROC = PROJ / "data" / "processed"
OUT = PROJ / "data" / "outputs"
FIG = PROJ / "artigo" / "figuras_en"

CAPACIDADE = 13946.0  # MWavg (max. observed ~ installed capacity)

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})


def _cap_line(ax):
    ax.axhline(CAPACIDADE, ls="--", color="gray", lw=1,
               label=f"Capacity ≈ {CAPACIDADE:.0f} MWavg")


def _caixa(ax, xy, w, h, texto, fc="#eaf1f8", ec="#1f6aa5", fs=10, bold=False):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=fs, weight="bold" if bold else "normal")


def _seta(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#444"))


def _baseline_linear_plato(df: pd.DataFrame, xs: np.ndarray | None = None):
    """Fits P(Q)=min(a+bQ, Pmax) by least squares and applies the plateau."""
    q = df.vazao.to_numpy()
    y = df.geracao.to_numpy()
    a, b = np.linalg.lstsq(np.vstack([np.ones_like(q), q]).T, y, rcond=None)[0]
    base_x = q if xs is None else xs
    pred = np.minimum(a + b * base_x, CAPACIDADE)
    return pred, a, b


def _pred_quadratica(df: pd.DataFrame, xs: np.ndarray):
    """Quadratic interpolation through three well-spaced observed points."""
    s = df.sort_values("vazao").reset_index(drop=True)
    n = len(s)
    idx = [0, n // 2, n - 1]
    (x0, y0), (x1, y1), (x2, y2) = [(s.vazao[i], s.geracao[i]) for i in idx]
    pred = (y0 * ((xs - x1) * (xs - x2)) / ((x0 - x1) * (x0 - x2))
            + y1 * ((xs - x0) * (xs - x2)) / ((x1 - x0) * (x1 - x2))
            + y2 * ((xs - x0) * (xs - x1)) / ((x2 - x0) * (x2 - x1)))
    a = (((y2 - y1) / (x2 - x1)) - ((y1 - y0) / (x1 - x0))) / (x2 - x0)
    b = (y1 - y0) / (x1 - x0) - a * (x0 + x1)
    vertice = -b / (2 * a)
    return pred, vertice


def fig1_conceitual():
    """Conceptual diagram: flow proxy -> physics-guided network -> reference curve."""
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    _caixa(ax, (0.2, 0.9), 2.2, 1.2, "Flow proxy\n(m³/s)\nPorto São José", fc="#eef7f1", ec="#2a8f6b")
    _caixa(ax, (3.6, 0.7), 2.8, 1.6,
           "Physics-guided\nnetwork\n(from scratch in Rust)\nP=ρgQHη as\nregularizer",
           fc="#eaf1f8", ec="#1f6aa5", bold=True)
    _caixa(ax, (7.5, 0.9), 2.3, 1.2, "Reference\ncurve\n(saturates at\ncapacity)",
           fc="#fdeeea", ec="#d1492f")
    _seta(ax, 2.45, 3.55, 1.5)
    _seta(ax, 6.45, 7.45, 1.5)
    ax.text(5.0, 0.25, "classical interpolation → spurious extrapolation in the saturation regime",
            ha="center", fontsize=8.5, style="italic", color="#777")
    fig.savefig(FIG / "fig1_conceitual.png")
    plt.close(fig)


def fig2_arquitetura():
    """MLP vs. physics-guided network: same graph, different losses and gradients."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    ink = "#111111"
    dark = "#333333"
    mid = "#666666"
    wire = "#bdbdbd"
    fill = "#f7f7f7"
    fill_dark = "#e9e9e9"

    def arrow(ax, xy0, xy1, color=dark, lw=1.2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(
            xy0, xy1, arrowstyle="-|>", mutation_scale=11, lw=lw,
            color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
            zorder=3,
        ))

    def elbow(ax, points, color=dark, lw=1.2, ls="-"):
        for p0, p1 in zip(points[:-2], points[1:-1]):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=color, lw=lw, linestyle=ls, solid_capstyle="round", zorder=2)
        arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls)

    def box(ax, xy, w, h, text, fs=8.5, bold=False, fc="white", hatch=None, lw=1.25):
        ax.add_patch(Rectangle(
            xy, w, h, facecolor=fc, edgecolor=ink, lw=lw, hatch=hatch, zorder=2
        ))
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
                fontsize=fs, weight="bold" if bold else "normal", color=ink, zorder=4)

    def network(ax, x0, y0, sx=1.0, sy=1.0):
        xs = [x0, x0 + 1.15 * sx, x0 + 2.30 * sx, x0 + 3.45 * sx]
        layers = [
            [y0],
            [y0 + 0.82 * sy, y0 + 0.27 * sy, y0 - 0.27 * sy, y0 - 0.82 * sy],
            [y0 + 0.82 * sy, y0 + 0.27 * sy, y0 - 0.27 * sy, y0 - 0.82 * sy],
            [y0],
        ]
        labels = ["input\n$q$", "hidden 1\n16 tanh", "hidden 2\n16 tanh", "output\nlinear"]
        nodes = []
        for idx, (xi, ys) in enumerate(zip(xs, layers)):
            layer_nodes = []
            for yi in ys:
                r = 0.15 * sx if idx in (0, 3) else 0.13 * sx
                ax.add_patch(Circle((xi, yi), r, fc="white", ec=ink, lw=1.35, zorder=3))
                layer_nodes.append((xi, yi))
            nodes.append(layer_nodes)
        for a, b in zip(nodes[:-1], nodes[1:]):
            for p0 in a:
                for p1 in b:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=wire, lw=0.75, zorder=1)
        ax.text(xs[0], y0, r"$q$", ha="center", va="center", fontsize=8.5, color=ink, zorder=4)
        ax.text(xs[-1], y0, r"$\hat{P}$", ha="center", va="center", fontsize=8.5, color=ink, zorder=4)
        # os rotulos em ingles sao mais largos que os originais: encolhe a fonte
        # quando o painel e comprimido (sx < 1) para nao colar "hidden 1"/"hidden 2".
        label_fs = min(7.3, 7.7 * sx)
        for xi, label in zip(xs, labels):
            ax.text(xi, y0 - 1.35 * sy, label, ha="center", va="top", fontsize=label_fs, color=dark)
        ax.text(x0 + 1.72 * sx, y0 + 1.40 * sy, r"MLP $1\to16\to16\to1$",
                ha="center", fontsize=8.1, color=dark, weight="bold")
        return {
            "out": (xs[-1] + 0.16 * sx, y0),
            "back": (x0 - 0.12 * sx, y0),
            "back_lane": x0 - 0.95 * sx,
        }

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(-0.65, 12.6)
        ax.set_ylim(0, 6.2)
        ax.axis("off")

    ax = axes[0]
    ax.text(6.3, 5.85, "Supervised MLP", ha="center", va="top", fontsize=13, weight="bold", color=ink)
    net = network(ax, 0.55, 3.60, sx=0.95, sy=0.92)
    box(ax, (5.05, 3.18), 1.30, 0.84, r"$\hat{P}$", fs=10.3, bold=True, fc=fill)
    box(ax, (7.20, 3.04), 1.85, 1.08,
        r"$\mathcal{L}_{data}$" "\n" "error", fs=8.8, fc=fill_dark)
    arrow(ax, net["out"], (5.05, 3.60), lw=1.25)
    arrow(ax, (6.35, 3.60), (7.20, 3.60), lw=1.25)
    elbow(ax, [(8.12, 3.04), (8.12, 1.18), (net["back_lane"], 1.18),
               (net["back_lane"], net["back"][1]), net["back"]],
          lw=1.25, ls="--")
    ax.text(4.25, 0.90, r"backprop: $\partial\mathcal{L}_{data}/\partial W$",
            ha="center", fontsize=8.1, color=dark)
    ax.text(6.3, 0.30, "Training driven only by the error between prediction and observed data.",
            ha="center", fontsize=8.2, color=mid)

    ax = axes[1]
    ax.text(6.3, 5.85, "Physics-guided network", ha="center", va="top", fontsize=13, weight="bold", color=ink)
    net = network(ax, 0.40, 3.60, sx=0.82, sy=0.86)
    box(ax, (3.90, 2.98), 1.60, 1.22,
        "evaluate at\n" r"$q\pm h,\ q$",
        fs=8.0, fc=fill)
    box(ax, (6.55, 4.44), 1.72, 0.78,
        r"$\mathcal{L}_{physics}$" "\n" "slope",
        fs=7.8, fc="white")
    box(ax, (6.55, 3.21), 1.72, 0.78,
        r"$\mathcal{L}_{data}$" "\n" "error",
        fs=7.8, fc=fill_dark)
    box(ax, (6.55, 1.98), 1.72, 0.78,
        r"$\mathcal{L}_{boundary}$" "\n" "cap",
        fs=7.8, fc="white")
    box(ax, (9.68, 3.06), 1.32, 1.06,
        r"$\mathcal{L}_{total}$", fs=9.6, bold=True, fc=fill_dark, lw=1.5)

    arrow(ax, net["out"], (3.90, 3.59), lw=1.25)
    elbow(ax, [(5.50, 3.94), (5.95, 3.94), (5.95, 4.83), (6.55, 4.83)], lw=1.15)
    arrow(ax, (5.50, 3.59), (6.55, 3.60), lw=1.15)
    elbow(ax, [(5.50, 3.24), (5.95, 3.24), (5.95, 2.37), (6.55, 2.37)], lw=1.15)

    elbow(ax, [(8.27, 4.83), (8.78, 4.83), (8.78, 3.86), (9.68, 3.86)], lw=1.15)
    arrow(ax, (8.27, 3.60), (9.68, 3.60), lw=1.15)
    elbow(ax, [(8.27, 2.37), (8.78, 2.37), (8.78, 3.32), (9.68, 3.32)], lw=1.15)

    elbow(ax, [(10.34, 3.06), (10.34, 1.03), (net["back_lane"], 1.03),
               (net["back_lane"], net["back"][1]), net["back"]],
          lw=1.25, ls="--")
    ax.text(5.75, 0.73, r"backprop: $\partial\mathcal{L}_{total}/\partial W$",
            ha="center", fontsize=8.1, color=dark)
    ax.text(6.3, 0.30, "The network is the same; the difference lies in the loss terms.",
            ha="center", fontsize=8.2, color=mid)

    fig.savefig(FIG / "fig2_arquitetura.png", dpi=240)
    plt.close(fig)


def fig3_ajuste_pinn():
    df = pd.read_csv(PROC / "dataset_final.csv", parse_dates=["data"])
    curva = pd.read_csv(OUT / "curva_otima.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    treino = df[df.data.dt.year <= 2022]
    teste = df[df.data.dt.year >= 2023]
    ax.scatter(treino.vazao, treino.geracao, s=6, alpha=0.20, color="#3b75af",
               label="Training (2015–2022)")
    ax.scatter(teste.vazao, teste.geracao, s=8, alpha=0.35, color="#d9872f",
               label="Temporal test (2023–2024)")
    ax.plot(curva.vazao, curva.geracao_otima, color="#d1492f", lw=2.4,
            label="1D physics-guided curve")
    _cap_line(ax)
    ax.set_xlabel("Flow proxy — Porto São José (m³/s)")
    ax.set_ylabel("Generation (MWavg)")
    ax.set_title("1D physics-guided network trained on 2015–2022")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIG / "fig3_ajuste_pinn.png")
    plt.close(fig)


def fig4_convergencia_loss():
    h = pd.read_csv(OUT / "historico_loss.csv")
    fis_col = "loss_fisica" if "loss_fisica" in h.columns else "loss_edp"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(h.epoca, h.loss_total, label="Total loss", lw=2)
    ax.plot(h.epoca, h.loss_dados, label="Data loss", lw=1.5)
    ax.plot(h.epoca, h[fis_col], label="Physics loss (slope)", lw=1.5)
    ax.plot(h.epoca, h.loss_contorno, label="Boundary loss", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Convergence of the 1D physics-guided network (λ = 1.0 / 1.0 / 1.0)")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig4_convergencia_loss.png")
    plt.close(fig)


def fig5_eficiencia():
    curva = pd.read_csv(OUT / "curva_otima.csv")
    # saturation knee: where generation reaches ~99% of capacity
    sat = curva[curva.geracao_otima >= 0.99 * CAPACIDADE]
    q_sat = sat.vazao.min() if not sat.empty else None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(curva.vazao, curva.eficiencia, color="#2a8f6b", lw=2.4,
            label="Relative productivity (MWavg per proxy m³/s)")
    if q_sat is not None:
        ax.axvline(q_sat, ls="--", color="#d1492f", lw=1.5,
                   label=f"Saturation ≈ {q_sat:.0f} m³/s")
    ax.set_xlabel("Flow proxy (m³/s)")
    ax.set_ylabel("Relative productivity (MWavg per proxy m³/s)")
    ax.set_title("Relative productivity and saturation point")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig5_eficiencia.png")
    plt.close(fig)


def fig6_retrospectiva():
    s = pd.read_csv(OUT / "predicoes_validacao_temporal.csv", parse_dates=["data"])
    teste = s[s.periodo == "teste"].copy()
    mensal = (teste.set_index("data")[
        [
            "geracao_real_mwmed",
            "persistencia_dia_anterior_mwmed",
            "persistencia_residual_mlp_mwmed",
            "rf_multivariada_mwmed",
            "mlp_multivariado_mwmed",
            "pinn_multivariada_mwmed",
        ]
    ].resample("ME").mean())
    modelos = [
        ("Persistence", "persistencia_dia_anterior_mwmed", "#666666"),
        ("Persist. + residual", "persistencia_residual_mlp_mwmed", "#7b61b8"),
        ("MLP w/o P(t-1)", "mlp_multivariado_mwmed", "#3b75af"),
        ("RF w/o P(t-1)", "rf_multivariada_mwmed", "#d9872f"),
        ("Physics-guided w/o P(t-1)", "pinn_multivariada_mwmed", "#2a8f6b"),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 6.8), constrained_layout=True)
    for rotulo, col, cor in modelos:
        residuo = mensal[col] - mensal.geracao_real_mwmed
        ax1.plot(mensal.index, residuo, lw=1.8, marker="o", ms=3.2,
                 label=rotulo, color=cor)
    ax1.axhline(0, color="black", lw=1.0)
    ax1.set_ylabel("Predicted - observed (MWavg)")
    ax1.set_title("Monthly residual on the temporal test")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax1.legend(fontsize=8.5, ncol=2)

    erros = [np.abs(teste[col] - teste.geracao_real_mwmed) for _, col, _ in modelos]
    bp = ax2.boxplot(erros, patch_artist=True, showmeans=True, widths=0.55)
    for patch, (_, _, cor) in zip(bp["boxes"], modelos):
        patch.set(facecolor=cor, alpha=0.35, edgecolor=cor)
    for mediana in bp["medians"]:
        mediana.set(color="black", lw=1.4)
    for media in bp["means"]:
        media.set(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4)
    ax2.set_xticklabels([m[0].replace(" ", "\n") for m in modelos])
    ax2.set_ylabel("Daily absolute error (MWavg)")
    ax2.set_title("Distribution of the daily absolute error")
    fig.suptitle("Out-of-sample temporal validation (2023–2024)")
    fig.savefig(FIG / "fig6_retrospectiva.png")
    plt.close(fig)


def fig7_ablacao():
    c = pd.read_csv(OUT / "curvas_ablacao.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(c.vazao, c.geracao_l0, color="#b03a2e", lw=2.2, ls="--",
            label="λ_physics = 0 (no physics residual) — does not saturate")
    ax.plot(c.vazao, c.geracao_l02, color="#2a8f6b", lw=2.4,
            label="λ_physics = 0.2 (robust final choice)")
    ax.plot(c.vazao, c.geracao_l1, color="#1f6aa5", lw=1.8, ls=":",
            label="λ_physics = 1.0 (point-estimate selection)")
    _cap_line(ax)
    ax.set_xlabel("Flow (m³/s)")
    ax.set_ylabel("Reference generation (MWavg)")
    ax.set_title("Ablation of the physics term: saturation of the reference curve")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.savefig(FIG / "fig7_ablacao.png")
    plt.close(fig)


def fig8_pinn_vs_interpolacao():
    """Comparison by machine-learning metrics on the temporal test."""
    m = pd.read_csv(OUT / "validacao_temporal.csv")
    teste = m[m.periodo == "teste_2023_2024"].copy()
    ordem = [
        "media_mensal_treino",
        "linear_com_plato",
        "persistencia_dia_anterior",
        "persistencia_residual_mlp",
        "rf_multivariada",
        "mlp_multivariado",
        "pinn_multivariada",
    ]
    rotulos = {
        "media_treino": "Mean",
        "media_mensal_treino": "Monthly mean",
        "interpolacao_quadratica": "Quadratic",
        "linear_com_plato": "Linear + plateau",
        "persistencia_dia_anterior": "Persistence",
        "persistencia_residual_mlp": "Persist. + residual",
        "mlp_1d": "1D MLP",
        "mlp_hidrologico": "Hydro MLP",
        "pinn_hidrologica": "Hydro physics-guided",
        "rf_hidrologica": "Hydro RF",
        "mlp_multivariado": "MLP w/o P(t-1)",
        "rf_multivariada": "RF w/o P(t-1)",
        "pinn_multivariada": "Physics-guided w/o P(t-1)",
    }
    teste = teste[teste.modelo.isin(ordem)].copy()
    teste["ordem"] = teste.modelo.map({k: i for i, k in enumerate(ordem)})
    teste = teste.sort_values("ordem")

    y = np.arange(len(teste))
    altura = 0.36
    fig, ax = plt.subplots(figsize=(6.9, 4.9))
    bars_rmse = ax.barh(y - altura / 2, teste.rmse_mwmed, altura, color="#3b75af")
    bars_mae = ax.barh(y + altura / 2, teste.mae_mwmed, altura, color="#d9872f")
    x_max = max(teste.rmse_mwmed.max(), teste.mae_mwmed.max()) + 640
    x_r2 = x_max - 35
    for nome, bars, valores in (
        ("RMSE", bars_rmse, teste.rmse_mwmed),
        ("MAE", bars_mae, teste.mae_mwmed),
    ):
        for bar, valor in zip(bars, valores):
            ax.text(
                valor + 24,
                bar.get_y() + bar.get_height() / 2,
                f"{nome} {valor:.0f}",
                ha="left",
                va="center",
                fontsize=7.4,
                color="#222",
            )
    for yi, r2 in zip(y, teste.r2):
        ax.text(
            x_r2,
            yi,
            f"R²={r2:.2f}",
            ha="right",
            va="center",
            fontsize=7.8,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4),
        )
    ax.set_yticks(y)
    ax.set_yticklabels([rotulos[m] for m in teste.modelo])
    ax.invert_yaxis()
    ax.set_xlabel("Test error (MWavg)")
    ax.set_title("Out-of-sample comparison (2023–2024)")
    ax.set_xlim(0, x_max)
    fig.savefig(FIG / "fig8_pinn_vs_interpolacao.png")
    plt.close(fig)


def fig9_vazao_turbinada_eq():
    """Diagnostic: equivalent turbined discharge inferred from P_obs/k."""
    d = pd.read_csv(OUT / "diagnostico_vazao_turbinada_equivalente.csv", parse_dates=["data"])
    resumo = pd.read_csv(OUT / "resumo_vazao_turbinada_equivalente.csv")
    d = d[d.periodo == "teste"].copy()
    mensal = d.set_index("data")[[
        "vazao_proxy_m3s",
        "vazao_turbinada_equivalente_m3s",
    ]].resample("ME").mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 5.6), constrained_layout=True)
    ax1.plot(mensal.index, mensal.vazao_proxy_m3s, marker="o", ms=3.2, lw=1.8,
             color="#3b75af", label="Porto São José")
    ax1.plot(mensal.index, mensal.vazao_turbinada_equivalente_m3s, marker="s", ms=3.0, lw=1.8,
             color="#2a8f6b", label=r"$Q_{eq}=P_{obs}/k$")
    ax1.set_ylabel("Monthly mean discharge (m³/s)")
    ax1.set_title("Flow proxy vs. equivalent turbined discharge")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax1.legend(fontsize=9)
    ax1.tick_params(labelsize=9)

    ax2.scatter(d.vazao_proxy_m3s, d.vazao_turbinada_equivalente_m3s,
                s=10, alpha=0.45, color="#6a4fb0")
    lim_min = min(d.vazao_proxy_m3s.min(), d.vazao_turbinada_equivalente_m3s.min())
    lim_max = max(d.vazao_proxy_m3s.max(), d.vazao_turbinada_equivalente_m3s.max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", color="#777", lw=1,
             label="1:1 line")
    corr = resumo.loc[resumo.periodo == "teste_2023_2024", "pearson_proxy_qeq"].iloc[0]
    ax2.set_xlabel("Porto São José discharge (m³/s)")
    ax2.set_ylabel(r"$Q_{eq}$ (m³/s)")
    ax2.set_title(f"Daily scatter on the test split (Pearson r={corr:.2f})")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=9)
    fig.savefig(FIG / "fig9_vazao_turbinada_eq.png")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    print("Rendering English figures into artigo/figuras_en/ ...")
    for nome, fn in [
        ("fig1_conceitual", fig1_conceitual),
        ("fig2_arquitetura", fig2_arquitetura),
        ("fig3_ajuste_pinn", fig3_ajuste_pinn),
        ("fig4_convergencia_loss", fig4_convergencia_loss),
        ("fig5_eficiencia", fig5_eficiencia),
        ("fig6_retrospectiva", fig6_retrospectiva),
        ("fig7_ablacao", fig7_ablacao),
        ("fig8_pinn_vs_interpolacao", fig8_pinn_vs_interpolacao),
        ("fig9_vazao_turbinada_eq", fig9_vazao_turbinada_eq),
    ]:
        fn()
        print(f"  ✓ {nome}.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
