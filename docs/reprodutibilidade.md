# Reprodutibilidade

Protocolo experimental, parâmetros e sementes que geram todos os números do artigo. Os comandos passo a passo estão no [README](../README.md#reproduzir-os-resultados).

## Repositório, licença e dados

- Código: https://github.com/davidogral/PGNN_itaipu, sob licença MIT (`LICENSE`).
- Dados: snapshot do ONS baixado em 03/06/2026 (licença CC-BY), versionado em `data/raw/` com somas SHA-256 em `data/raw/SHA256SUMS`. Os dados processados (`data/processed/`) e todas as saídas (`data/outputs/`) também estão versionados.

## Comandos principais

```bash
cd rust
cargo test --release
cargo run --release
cd ..
python scripts/python/verificar_reprodutibilidade.py
python scripts/python/teste_bootstrap_residual_persistencia.py
python scripts/python/benchmark_python.py --pure-epochs 10 --numpy-epochs 2000
python scripts/python/gerar_graficos.py
```

## Verificação registrada

Em 15/09/2026, a partir de um clone limpo do repositório, num Apple M5 (macOS arm64) com Rust 1.96.0 e Python 3.9.6:

- as 36 somas SHA-256 de `data/raw/` conferiram e os 18 testes passaram;
- o ETL (`limpar_dados.py`, `cruzar_datasets.py`, `exportar_para_rust.py`) recriou os seis arquivos de `data/processed/` byte a byte, em cerca de 3 s;
- `cargo run --release` (de 7 a 14 min), o bootstrap (cerca de 23 s) e `gerar_graficos.py` regeneraram `data/outputs/`, e `verificar_reprodutibilidade.py` confirmou 42 de 42 arquivos: todas as métricas, curvas e predições foram reproduzidas, e só as colunas de tempo de execução (`tempo_*`) diferiram.

## Protocolo experimental

- Treino principal: 2015–2022 (`n=2919`).
- Teste temporal: 2023–2024 (`n=731`).
- Seleção interna de `lambda_f`: treino 2015–2020, validação 2021–2022.
- Candidatos de `lambda_f`: `0.0`, `0.05`, `0.1`, `0.2`, `0.5`, `1.0`.
- Escolha final de `lambda_f`: `0.2`. A varredura pontual com semente fixa seleciona `1.0`, mas a sensibilidade com 10 sementes favorece `0.2` (`1540±3` MWmed contra `1563±72` MWmed); por isso o artigo não reivindica superioridade geral de `1.0`.
- Normalização: min-max ajustado apenas no treino de cada corte temporal.
- Validação móvel: teste anual 2020–2024, sempre treinando com os anos anteriores.

## Parâmetros numéricos

- Arquitetura univariada: `1 -> 16 -> 16 -> 1`.
- Arquitetura multivariada: `5 -> 16 -> 16 -> 1`.
- Entradas multivariadas treináveis: `vazao_norm`, `ena_bruta_norm`, `mes_sin`, `mes_cos`, `vazao_lag1_norm`.
- `geracao_lag1` fica no dataset apenas para o baseline de persistência, não como entrada dos modelos treináveis.
- Modelo residual: aprende `(geracao - geracao_lag1) / delta_geracao` com as mesmas entradas multivariadas; a previsão final é `geracao_lag1 + alpha * residuo_predito`.
- Seleção de `alpha` residual: grade `0.00..1.00` com passo `0.05`, validação 2021–2022; valor escolhido: `0.30`.
- Vazão turbinada equivalente (diagnóstico): `Q_eq = P_real / k`, com `k = rho*g*H*eta`; não é usada como entrada de treino.
- Ativação: `tanh` nas camadas ocultas e saída linear.
- Otimizador: Adam manual, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.
- Taxa de aprendizado: `0.01`.
- Épocas principais: `2000`.
- Épocas na seleção interna: `1200`.
- Épocas nas janelas móveis: `600`.
- Épocas no cenário operacional autorregressivo: `1200`.
- Passo de diferenças finitas da perda física: `H_FD = 1e-3`.
- Passo de diagnóstico físico: `H_FD_DIAGNOSTICO = 1e-3`.
- Pontos físicos no treino final: as próprias amostras do full-batch (`n=2919`).
- Grade de diagnóstico físico: 100 pontos uniformes em `q_norm in [0, 1]`.

## Sementes fixas

- Seleção de `lambda_f`, ablação e rede física 1D: `7`.
- MLP 1D: `13`.
- MLP hidrológico: `19`.
- MLP multivariado: `17`.
- MLP residual para seleção de `alpha`: `41`.
- MLP residual final: `43`.
- Rede física hidrológica: `23`.
- Rede física multivariada: `11`.
- Random Forest hidrológica: `31`.
- Random Forest multivariada: `37`.
- Janelas móveis: MLP `200 + ano`, rede física `300 + ano`, residual interno `400 + ano`, residual final `500 + ano`, RF `101 + ano`.
- Sensibilidade multissemente de `lambda_f`: `7`, `17`, `31`, `43`, `59`, `73`, `89`, `101`, `131`, `151`.
- Bootstrap em blocos: semente `20260811`, 20.000 reamostragens, blocos de 7, 14 e 30 dias.

## Artefatos principais

- `data/outputs/validacao_temporal.csv`
- `data/outputs/predicoes_validacao_temporal.csv`
- `data/outputs/validacao_janelas.csv`
- `data/outputs/validacao_movel_operacional.csv`
- `data/outputs/bootstrap_residual_persistencia.csv`
- `data/outputs/sensibilidade_lambda_sementes.csv`
- `data/outputs/diagnostico_fisico.csv`
- `data/outputs/curvas_diagnostico_fisico.csv`
- `data/outputs/selecao_residual_persistencia.csv`
- `data/outputs/diagnostico_vazao_turbinada_equivalente.csv`
- `data/outputs/resumo_vazao_turbinada_equivalente.csv`
- `data/outputs/benchmark_comparativo.csv`
- Figuras do artigo: geradas localmente por `scripts/python/gerar_graficos.py` a partir dos CSVs acima.
