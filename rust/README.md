# Núcleo Rust

Implementação do zero — sem bibliotecas de aprendizado de máquina nem diferenciação automática — da rede neural guiada por física (PGNN), dos baselines, da validação temporal e da análise retrospectiva. As únicas dependências são `serde`, `serde_json` e `csv`, com versões fixadas em `Cargo.lock`. O porquê de cada escolha está em [`../docs/decisoes_tecnicas.md`](../docs/decisoes_tecnicas.md).

## Executar

```bash
cargo run --release     # pipeline completo: lê ../data/processed/ e grava ../data/outputs/
cargo test --release    # 18 testes unitários, incluindo o gradient check do backprop
```

Os caminhos de entrada e saída são resolvidos a partir deste diretório (`CARGO_MANIFEST_DIR`), então o binário funciona a partir de qualquer pasta. A CI verifica cada push com `cargo fmt --check`, `cargo clippy -- -D warnings` e `cargo test`.

## O que o binário executa

1. **Fase 3 — métodos clássicos:** interpolação quadrática (formas de Lagrange e de Newton) e Newton-Raphson, usados como baseline e para localizar extremos.
2. **Fase 4 — sanidade da rede:** ajuste de `y = x²` para validar forward, backprop e Adam antes de adicionar a física.
3. **Fase 5 — rede guiada por física e validação:** seleção interna de `λ_f`, ablação e sensibilidade a sementes; redes física 1D, hidrológica e multivariada; MLPs, Random Forest, persistência e modelo residual; diagnóstico físico; validação temporal (treino 2015–2022, teste 2023–2024) e móvel (2020–2024); exploração temporal/residual e cenário operacional (`experimentos.rs`).
4. **Fase 6 — análise retrospectiva:** geração real × curva de referência, com resíduos separados por período de treino e de teste.

## Saídas em `data/outputs/`

| Tema | Arquivos |
|---|---|
| Seleção e ablação de `λ_f` | `selecao_lambda.csv`, `ablacao_lambda.csv`, `curvas_ablacao.csv`, `sensibilidade_lambda_sementes.csv` |
| Validação temporal | `validacao_temporal.csv`, `predicoes_validacao_temporal.csv`, `selecao_residual_persistencia.csv` |
| Validação móvel anual | `validacao_janelas.csv`, `validacao_movel_operacional.csv` |
| Curvas de referência | `curva_otima.csv`, `curva_multivariada.csv`, `curva_pinn_hidrologica.csv`, `curva_mlp_hidrologico.csv`, `curva_mlp_multivariado.csv` |
| Convergência | `historico_loss*.csv` |
| Diagnóstico físico | `diagnostico_fisico.csv`, `curvas_diagnostico_fisico.csv`, `diagnostico_vazao_turbinada_equivalente.csv`, `resumo_vazao_turbinada_equivalente.csv` |
| Exploração temporal/residual | `exploracao_temporal_residual.csv`, `exploracao_features_temporais.csv`, `exploracao_pesos_lineares.csv`, `exploracao_fator_efetivo.csv`, `predicoes_exploracao_temporal.csv` |
| Retrospectiva | `simulacao_retrospectiva.csv` |
| Tempo de execução | `benchmark_rust.csv` |

Os demais CSVs da pasta vêm dos scripts Python (bootstrap, benchmark e tabela comparativa); veja [`../scripts/README.md`](../scripts/README.md).

## Organização de `src/`

```text
src/
├── main.rs                 orquestra as fases 3 a 6 e grava as saídas
├── experimentos.rs         defasagens hidrológicas, modelos residuais e cenário operacional
├── data/
│   └── loader.rs           leitura de dataset_final.csv
├── math/
│   ├── interpolacao.rs     interpolação quadrática (Lagrange e Newton)
│   └── newton_raphson.rs   raízes e extremos 1D
├── neural/
│   ├── network.rs          camadas, pesos e vieses (init Xavier com PRNG próprio)
│   ├── forward.rs          forward pass com tanh
│   ├── backward.rs         backpropagation manual
│   ├── optimizer.rs        Adam
│   └── loss.rs             perdas de dados, física (inclinação e saturação) e contorno
├── ml/
│   └── arvores.rs          Random Forest para regressão
├── pinn/
│   ├── treinamento.rs      laço de treino com a perda física
│   └── otimizacao.rs       curva de referência pós-treino, limitada a [0, Pmax]
└── simulacao/
    └── retrospectiva.rs    real × referência, resíduos por período
```

O módulo `pinn/` mantém o nome da primeira fase do projeto e implementa a rede guiada por física descrita no artigo.
