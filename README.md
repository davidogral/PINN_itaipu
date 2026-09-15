<div align="center">

# PGNN-Itaipu

**Rede neural guiada por física, implementada do zero em Rust, para modelar a geração diária da Usina Hidrelétrica de Itaipu**

[![CI](https://github.com/davidogral/PGNN_itaipu/actions/workflows/ci.yml/badge.svg)](https://github.com/davidogral/PGNN_itaipu/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)](LICENSE)
[![Rust 1.96](https://img.shields.io/badge/Rust-1.96-orange.svg?logo=rust)](rust/)
[![Dados: ONS, CC-BY](https://img.shields.io/badge/dados-ONS%20%C2%B7%20CC--BY-2e7d32.svg)](data/raw/README.md)
[![Citar](https://img.shields.io/badge/citar-CITATION.cff-6f42c1.svg)](CITATION.cff)

Código, dados e resultados do artigo<br>
*Modelagem da Geração Hidrelétrica de Itaipu via Redes Neurais Guiadas por Física em Rust: Validação Temporal, Persistência e Previsão Operacional*<br>
aprovado no **Latin.Science 2026** (Latinoware 2026).

</div>

---

## Sobre

O projeto modela a relação entre a vazão do Rio Paraná no posto Porto São José, usada como *proxy* da disponibilidade hídrica a montante, e a geração diária de Itaipu. O modelo é uma **PGNN** (*physics-guided neural network*, rede neural guiada por física): inspirada em PINNs, mas sem resolver uma equação diferencial. A lei de potência hidráulica $P = \rho g Q H \eta$ entra na perda como regularizador da inclinação e da saturação na capacidade, e não como balanço ponto a ponto, porque a vazão disponível não é a vazão turbinada.

Toda a parte numérica — forward pass, backpropagation, Adam, perdas com física, Random Forest e baselines — foi escrita **do zero em Rust, sem bibliotecas de aprendizado de máquina nem diferenciação automática**. As únicas dependências são `serde`, `serde_json` e `csv`, e todo gradiente é explícito e auditável.

**O que o trabalho mostra**

- **Menor erro nos dados não significa modelo válido.** Sem o termo físico, a rede tem o menor erro de treino, mas a curva não satura e chega a prever menos geração com mais água.
- **A persistência é um baseline muito forte.** Repetir a geração do dia anterior atinge RMSE de 590 MWmed no teste 2023–2024. As *proxies* hidrológicas públicas acrescentam um sinal pequeno, porém estatisticamente positivo: 4,49 MWmed de redução (IC95%: 1,05 a 7,74).
- **A memória operacional é a rota preditiva mais promissora.** Com geração passada como entrada, um MLP residual reduz o RMSE médio de 596 para 529 MWmed na validação móvel 2020–2024 e vence a persistência em todos os anos.

## Principais resultados

Treino em 2015–2022 e teste em 2023–2024, com normalização ajustada só no treino.

| Modelo | RMSE (MWmed) | MAE (MWmed) | R² |
|---|---:|---:|---:|
| Persistência, $\hat P_t = P_{t-1}$ | 590 | 465 | 0,88 |
| Persistência + MLP residual hidrológico ($\alpha = 0{,}30$) | 586 | 460 | 0,88 |
| **MLP residual operacional** (geração passada + hidrologia defasada) | **508** | **391** | **0,91** |
| MLP univariado | 1.487 | 1.213 | 0,23 |
| Random Forest sem $P_{t-1}$ | 1.529 | 1.192 | 0,19 |
| Rede física multivariada sem $P_{t-1}$ | 1.672 | 1.263 | 0,03 |
| Interpolação quadrática | 1.690 | 1.339 | 0,01 |
| Rede física univariada ($\lambda_f = 0{,}2$), com saturação | 1.776 | 1.341 | −0,10 |

As tabelas completas, a ablação de $\lambda_f$, a análise multissemente e a validação móvel estão em [`docs/decisoes_tecnicas.md`](docs/decisoes_tecnicas.md) e em [`data/outputs/`](data/outputs/).

## Reproduzir os resultados

Todos os dados — do snapshot bruto do ONS às saídas finais — estão versionados. Por isso, depois de rodar o pipeline, **um único comando confere se cada arquivo publicado foi reproduzido**.

**Requisitos:** [Rust](https://rustup.rs) (testado com 1.96.0) e, para as etapas em Python, Python 3.9 ou mais recente (testado com 3.9.6).

### 1. Treinar todos os modelos

```bash
git clone https://github.com/davidogral/PGNN_itaipu.git
cd PGNN_itaipu/rust
cargo test --release    # 18 testes, incluindo o gradient check do backprop
cargo run --release     # treina tudo e regrava data/outputs/ (7 a 14 min num Apple M5)
cd ..
python3 scripts/python/verificar_reprodutibilidade.py   # compara com os arquivos versionados
```

O verificador usa só a biblioteca padrão do Python. Ele compara cada CSV com a versão do commit, célula a célula, e ignora apenas as colunas de tempo de execução (`tempo_*`), que mudam a cada rodada.

### 2. Refazer o pré-processamento a partir dos dados brutos

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/python/limpar_dados.py        # filtra os postos/bacia e agrega a geração por dia
python scripts/python/cruzar_datasets.py     # junta as séries por data
python scripts/python/exportar_para_rust.py  # normaliza e grava data/processed/dataset_final.csv
```

A partir do snapshot em `data/raw/`, esses passos recriam `data/processed/` byte a byte. Para baixar os dados de novo do portal do ONS, use `python scripts/python/download_datasets.py 2000 2024` e compare com [`data/raw/SHA256SUMS`](data/raw/SHA256SUMS): o ONS pode revisar séries antigas.

### 3. Análises complementares

```bash
python scripts/python/teste_bootstrap_residual_persistencia.py  # IC95% do ganho do residual (bootstrap em blocos de 14 dias)
python scripts/python/benchmark_python.py --pure-epochs 10 --numpy-epochs 2000  # Rust × Python puro × NumPy
python scripts/python/gerar_graficos.py                         # figuras do artigo, geradas localmente em artigo/figuras/
```

**Determinismo.** O gerador pseudoaleatório é próprio (*xorshift64\**) e as sementes são fixas por modelo (lista em [`docs/reprodutibilidade.md`](docs/reprodutibilidade.md)). A partir de um clone limpo, o pipeline completo reproduziu todas as métricas, curvas e predições publicadas byte a byte (macOS arm64, Rust 1.96.0); só os tempos de execução variaram. Em outra plataforma pode haver diferença na última casa decimal, porque funções como `tanh` e `exp` vêm da biblioteca matemática do sistema.

## Estrutura do repositório

```text
PGNN_itaipu/
├── rust/                       núcleo numérico (Rust, sem bibliotecas de ML)
│   └── src/
│       ├── main.rs             orquestra as fases 3 a 6
│       ├── experimentos.rs     exploração temporal/residual e cenário operacional
│       ├── neural/             rede, forward, backprop, Adam e perdas com física
│       ├── pinn/               treino regularizado e curva de referência
│       ├── ml/                 Random Forest
│       ├── math/               interpolação quadrática e Newton-Raphson
│       ├── data/               leitura do dataset
│       └── simulacao/          análise retrospectiva
├── scripts/python/             ETL dos dados do ONS, figuras, benchmark e bootstrap
├── data/
│   ├── raw/                    snapshot dos dados do ONS (03/06/2026) + SHA256SUMS
│   ├── processed/              séries diárias cruzadas e normalizadas (entrada do Rust)
│   └── outputs/                métricas, curvas, históricos de perda e predições
├── docs/                       decisões técnicas, protocolo de reprodutibilidade e glossário
├── requirements.txt            dependências Python com versões fixadas
└── CITATION.cff                metadados de citação
```

As fases seguem a ordem do desenvolvimento: **1–2** dados e ETL (Python), **3** métodos clássicos, **4** sanidade da rede em $y = x^2$, **5** rede guiada por física, baselines e validação temporal, **6** análise retrospectiva.

> **Sobre o nome.** O projeto começou como "PINN-Itaipu" e passou a se chamar PGNN-Itaipu para refletir o enquadramento do artigo. Identificadores internos daquela fase foram mantidos para não alterar os artefatos publicados: o módulo `rust/src/pinn/` e os rótulos `pinn_*` nos CSVs designam a rede guiada por física.

## Dados

Três conjuntos públicos do [Portal de Dados Abertos do ONS](https://dados.ons.org.br), publicados sob licença **Creative Commons Atribuição (CC-BY)** e baixados em **3 de junho de 2026**:

| Conjunto | Granularidade | Uso no projeto |
|---|---|---|
| [Geração de Itaipu](https://dados.ons.org.br/dataset/geracao_itaipu) | horária | alvo, agregado em média diária (MWmed) |
| [Grandezas fluviométricas](https://dados.ons.org.br/dataset/grandezas_fluviometricas) | diária | vazão do posto Porto São José (Rio Paraná), média dos dois medidores |
| [ENA diária por bacia](https://dados.ons.org.br/dataset/ena-diario-por-bacia) | diária | Energia Natural Afluente bruta da bacia do Paraná |

O artigo considera os registros de janeiro de 2000 a dezembro de 2024 para geração e ENA e de janeiro de 2015 a dezembro de 2024 para a vazão. O modelo usa a janela comum de 01/01/2015 a 31/12/2024 (3.651 dias). Colunas, filtros e cuidados com os identificadores de postos estão em [`data/raw/README.md`](data/raw/README.md).

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/decisoes_tecnicas.md`](docs/decisoes_tecnicas.md) | cada decisão de modelagem e o porquê: Rust sem ML, `tanh`, Adam, perda física, escolha da vazão, seleção de $\lambda$ |
| [`docs/reprodutibilidade.md`](docs/reprodutibilidade.md) | protocolo experimental, hiperparâmetros, sementes e artefatos |
| [`docs/exploracao_plato_temporal_residual.md`](docs/exploracao_plato_temporal_residual.md) | defasagens hidrológicas, modelos residuais e cenário operacional |
| [`docs/glossario.md`](docs/glossario.md) | termos técnicos em linguagem simples |
| [`docs/referencias_leitura.md`](docs/referencias_leitura.md) | leituras recomendadas |
| [`rust/README.md`](rust/README.md) · [`scripts/README.md`](scripts/README.md) | detalhes de cada parte do código |

## Como citar

O GitHub gera a citação a partir de [`CITATION.cff`](CITATION.cff) (botão *Cite this repository*). Em BibTeX:

```bibtex
@inproceedings{specia2026itaipu,
  author    = {Davi Specia and Luiz Henrique Zavatini Feltrin and Tainá Dreissig and
               Miguel Mantoan Castellani and Vinícius Carraro and Wagner Wilson Ávila Bombardelli},
  title     = {Modelagem da Geração Hidrelétrica de Itaipu via Redes Neurais Guiadas por Física
               em Rust: Validação Temporal, Persistência e Previsão Operacional},
  booktitle = {Latin.Science 2026 -- Latinoware 2026, 23º Congresso Latino-americano de
               Software Livre e Tecnologias Abertas},
  year      = {2026},
  url       = {https://github.com/davidogral/PGNN_itaipu}
}
```

## Autores

Davi Specia, Luiz Henrique Zavatini Feltrin, Tainá Dreissig, Miguel Mantoan Castellani e Vinícius Carraro, estudantes de graduação em Inteligência Artificial, com orientação de Wagner Wilson Ávila Bombardelli — **Faculdade Donaduzzi**, Toledo, Paraná.

## Licença

Código sob licença [MIT](LICENSE). Os dados em `data/raw/` são do Operador Nacional do Sistema Elétrico (ONS), redistribuídos sob a licença CC-BY da fonte. Os dados derivados em `data/processed/` e `data/outputs/` mantêm essa atribuição.

## Agradecimentos

Ao ONS, pelos dados públicos que viabilizaram este estudo, e à Faculdade Donaduzzi, pelo apoio ao projeto.

<details>
<summary><strong>English summary</strong></summary>

<br>

This repository contains the code, data and results of a paper accepted at **Latin.Science 2026** (Latinoware 2026). It models the relationship between an upstream Paraná River discharge proxy (Porto São José gauging station) and the daily generation of the Itaipu hydropower plant with a **physics-guided neural network written from scratch in Rust**, with no machine-learning or automatic-differentiation libraries. The hydraulic power law is used as a regularizer for the marginal slope and for saturation at plant capacity.

Key findings: minimizing the data loss alone yields a physically invalid curve; day-ahead persistence is a very strong baseline (RMSE 590 MWavg on 2023–2024); public hydrological proxies add a small but statistically positive signal; and an operational autoregressive residual MLP lowers the 2020–2024 rolling RMSE from 596 to 529 MWavg.

To reproduce every result, run `cargo run --release` inside `rust/`. All data, from the raw ONS snapshot to the final outputs, is versioned, so `git status` shows whether a run reproduced the published files exactly.

</details>
