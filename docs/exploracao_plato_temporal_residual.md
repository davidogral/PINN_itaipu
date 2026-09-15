# Exploracao do Plato Temporal e Residual

Rodada executada em 2026-08-11 para testar caminhos de melhora sem vazamento de informacao.

## Objetivo

Separar tres perguntas que estavam misturadas:

- **Previsao operacional:** o modelo usa somente variaveis disponiveis antes do dia previsto.
- **Residual sobre persistencia:** as variaveis hidrologicas explicam a mudanca de geracao de ontem para hoje?
- **Fisica explicativa:** a relacao hidraulica continua sendo usada como regularizacao/diagnostico, nao como promessa de melhor preditor diario com vazao proxy.

## O que foi implementado

Nova fase no Rust: `rust/src/experimentos.rs`, chamada a partir da Fase 5.

Artefatos gerados:

- `data/outputs/exploracao_temporal_residual.csv`
- `data/outputs/exploracao_features_temporais.csv`
- `data/outputs/exploracao_pesos_lineares.csv`
- `data/outputs/exploracao_fator_efetivo.csv`
- `data/outputs/predicoes_exploracao_temporal.csv`
- `data/outputs/validacao_movel_operacional.csv`

Os experimentos usam treino 2015-2022 e teste 2023-2024. Como as janelas usam ate 30 dias de historico, o treino efetivo cai de 2919 para 2889 amostras; o teste permanece com 731 dias.

## Status das ideias levantadas

| Ideia | Status nesta rodada | Evidência/saída |
|---|---|---|
| Mais defasagens hidrológicas | Implementada | `hidro_lags_medias_30d` em `exploracao_temporal_residual.csv` |
| Prever anomalia ou residual | Implementada | formulações `anomalia_mensal` e `residuo_sobre_persistencia` |
| Baselines fortes para residual | Implementada | persistência, delta mensal, ridge, RF e MLP residual |
| Atenção/janela temporal simples | Implementada como MLP tabular com janela | MLP com 26 ou 33 entradas defasadas, sem recorrência pesada |
| Aprender fator efetivo | Implementada em versão linear | `exploracao_fator_efetivo.csv` |
| Física comportamental em vez de inclinação fixa | Diagnosticada e documentada como próxima alteração de loss | a evidência atual favorece monotonicidade/saturação/suavidade quando a entrada é proxy |
| Validação móvel do melhor cenário operacional | Implementada | `validacao_movel_operacional.csv` |

## Features testadas

### Hidrologia basica defasada

Usa apenas:

- mes_sin, mes_cos
- vazao t-1
- ENA t-1

### Hidrologia longa

Usa somente `t-1` ou anterior:

- vazao t-1, t-2, t-3, t-7, t-14, t-30
- medias moveis de vazao de 3, 7, 14 e 30 dias
- variacoes de vazao: `q_lag1 - q_lag2` e `q_lag1 - q_lag7`
- ENA t-1, t-2, t-3, t-7, t-14, t-30
- medias moveis de ENA de 3, 7, 14 e 30 dias
- variacoes de ENA: `ena_lag1 - ena_lag2` e `ena_lag1 - ena_lag7`
- sazonalidade mensal

### Operacional com geracao passada

Inclui a hidrologia longa e adiciona:

- geracao t-1, t-2, t-7
- medias moveis de geracao de 3 e 7 dias
- variacoes de geracao: `p_lag1 - p_lag2` e `p_lag1 - p_lag7`

Esse grupo continua operacionalmente justo, pois usa apenas passado, mas deve ser interpretado como modelo autorregressivo.

## Alvos testados

- **Geração absoluta:** prever `P_t`.
- **Anomalia mensal:** prever `P_t - media_mensal_historica(mes)`.
- **Residual sobre persistência:** prever `P_t - P_{t-1}`.

Cada formulação foi comparada com:

- regressão linear ridge residual/absoluta
- Random Forest em Rust
- MLP com janela temporal simples em Rust

Também foi incluído o baseline `P_t = P_{t-1} + media_mensal(P_t-P_{t-1})`.

## Resultados principais no teste 2023-2024

| Modelo | Features | Formulação | RMSE | MAE | R² |
|---|---|---|---:|---:|---:|
| Persistência | n/a | baseline | 590.44 | 464.56 | 0.8789 |
| Persistência + delta mensal | n/a | residual | 590.39 | 464.00 | 0.8789 |
| MLP residual | hidrologia longa | `P_t-P_{t-1}` | 578.77 | 452.66 | 0.8836 |
| Ridge residual | hidrologia longa | `P_t-P_{t-1}` | 580.10 | 454.35 | 0.8831 |
| RF residual | hidrologia longa | `P_t-P_{t-1}` | 587.09 | 462.79 | 0.8802 |
| MLP residual | operacional + geração passada | `P_t-P_{t-1}` | **507.55** | **390.86** | **0.9105** |
| MLP absoluto | operacional + geração passada | `P_t` | 509.06 | 395.61 | 0.9100 |
| RF residual | operacional + geração passada | `P_t-P_{t-1}` | 512.39 | 395.60 | 0.9088 |
| Ridge residual | operacional + geração passada | `P_t-P_{t-1}` | 516.08 | 399.39 | 0.9075 |

Leitura:

- Lags e médias hidrológicas sozinhos melhoram a persistência de 590.44 para 578.77 MWmed, cerca de 2.0%. É um ganho real, mas pequeno.
- A formulação residual é a melhor forma de extrair esse sinal hidrológico. Modelos absolutos/anomalia com apenas hidrologia continuam muito acima de 1300 MWmed.
- O salto relevante vem ao aceitar geração passada como entrada operacional: melhor RMSE de 507.55 MWmed, cerca de 14.0% abaixo da persistência.
- Como `P_{t-1}` e estatísticas recentes de geração carregam quase toda a dinâmica de curto prazo, esse resultado deve ser apresentado como modelo operacional autorregressivo, não como prova de causalidade hidrológica.

## Validação móvel do cenário operacional

Para testar se o melhor cenário do split 2023-2024 era robusto, foi executada uma validação móvel anual específica com o grupo `operacional_lags_com_geracao`. Em cada ano de teste, o modelo foi treinado apenas com anos anteriores:

| Ano | Persistência RMSE | MLP residual operacional RMSE | Melhora |
|---:|---:|---:|---:|
| 2020 | 531.57 | 478.42 | 10.00% |
| 2021 | 579.74 | 568.88 | 1.87% |
| 2022 | 702.92 | 584.84 | 16.80% |
| 2023 | 680.76 | 557.90 | 18.05% |
| 2024 | 483.86 | 455.42 | 5.88% |

Médias 2020-2024:

| Modelo | RMSE médio | MAE médio | R² médio | Melhora média vs. persistência |
|---|---:|---:|---:|---:|
| Persistência | 595.77 | 473.52 | 0.8372 | 0.00% |
| Ridge residual operacional | 535.29 | 421.73 | 0.8719 | 9.44% |
| RF residual operacional | 533.35 | 423.09 | 0.8729 | 10.10% |
| MLP residual operacional | **529.09** | **416.65** | **0.8754** | **11.19%** |
| MLP absoluto operacional | 569.59 | 449.90 | 0.8574 | 2.86% |

Leitura para o artigo: este passa a ser o resultado preditivo mais forte, mas deve entrar como uma pergunta separada. Sem geração passada como entrada, o artigo mede o sinal hidrológico incremental; com geração passada, mede previsão operacional autorregressiva justa, pois todas as entradas são observações de `t-1` ou anteriores.

## Fator efetivo aprendido

Foram ajustados modelos simples `P ~= a + k_efetivo Q` com variáveis defasadas:

| Variável | `k_efetivo` MW/(m³/s) | Razão vs. `k_fisico` | RMSE teste |
|---|---:|---:|---:|
| q_lag1 | 0.5672 | 0.544 | 1759.22 |
| q_ma7 | 0.6553 | 0.629 | 1703.78 |
| q_ma30 | 0.7787 | 0.747 | 1679.91 |

O fator efetivo fica abaixo do `k_fisico ~= 1.0418 MW/(m³/s)`, o que reforça que Porto São José é proxy, não vazão turbinada. A média móvel de 30 dias melhora a relação linear simples, mas ainda fica longe da persistência.

## Conclusão desta rodada

A saída mais promissora para previsão diária não é "mais física forte" sobre a vazão proxy. A rota mais defensável é:

1. Manter a física como regularização/diagnóstico e explicitar a limitação da proxy.
2. Usar modelos residuais quando o objetivo for medir sinal hidrológico incremental.
3. Para previsão operacional, aceitar geração passada como entrada e declarar o modelo como autorregressivo operacional.
4. Se a intenção for contribuição física mais forte, buscar vazão turbinada, vertimento, queda líquida e nível do reservatório; sem isso, usar penalizações comportamentais (monotonicidade, saturação e suavidade) é mais honesto do que impor inclinação fixa.

O RMSE de 507.55 MWmed no split 2023-2024 agora tem suporte adicional da validação móvel anual: o MLP residual operacional venceu a persistência em todos os anos de 2020 a 2024. Ainda assim, ele deve ser apresentado como resultado operacional autorregressivo, não como substituto da contribuição física.
