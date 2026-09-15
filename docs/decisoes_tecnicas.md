# Decisões Técnicas

Registro das principais escolhas de arquitetura e modelagem do projeto, com a justificativa de cada uma. Serve para documentar o raciocínio por trás dos resultados e para orientar quem continuar o desenvolvimento.

---

## 1. Por que Rust, e sem bibliotecas de Machine Learning?

- **Objetivo didático/científico:** implementar a PINN "do zero" demonstra domínio real de forward pass, backpropagation e otimização — diferencial num short paper de graduação.
- **Transparência:** sem `tch`, `candle`, `burn` ou autodiff, todo gradiente é explícito e auditável. Não há "caixa-preta" — cada derivada está no código.
- **Desempenho e segurança:** Rust dá performance de baixo nível (laços de treino rápidos) com segurança de memória, sem coletor de lixo interferindo no tempo de treino.
- **Reprodutibilidade:** binário único, dependências mínimas (`serde`, `csv`, `serde_json`), fácil de compilar em qualquer máquina.
- **Contraste com Python:** Python fica restrito a ETL e gráficos, deixando claro que o "núcleo científico" é a implementação manual em Rust.

> Restrição assumida: **nenhuma** lib de ML ou de diferenciação automática. Apenas serialização de dados.

---

## 2. Por que `tanh` como função de ativação?

- **Suavidade (C∞):** PINNs exigem derivadas da saída em relação às entradas para montar o resíduo físico. `tanh` é infinitamente diferenciável, então as derivadas necessárias existem e são contínuas — diferente de `ReLU`, cuja derivada pode ser descontínua.
- **Saída centrada em zero:** `tanh ∈ (−1, 1)`, o que ajuda a convergência em redes pequenas e combina bem com entradas normalizadas.
- **Derivada barata:** `tanh'(x) = 1 − tanh²(x)`, fácil de implementar no backprop manual.
- **Padrão na literatura de PINNs:** Raissi et al. (2019) e a maioria dos trabalhos seguintes usam `tanh` exatamente pela necessidade de diferenciabilidade.

---

## 3. Por que Adam como otimizador?

- **Taxa de aprendizado adaptativa por parâmetro:** combina momentum (1ª ordem) com escala por variância (2ª ordem), acelerando vales rasos e amortecendo direções ruidosas.
- **Robustez em loss multiobjetivo:** a loss da PINN soma termos de escalas diferentes (dados + física + contorno); Adam lida melhor com gradientes heterogêneos do que SGD puro.
- **Pouca sintonia:** funciona bem com hiperparâmetros padrão `β1=0.9`, `β2=0.999`, `ε=1e-8`.
- **Implementável do zero:** apenas dois vetores de estado (`m`, `v`) por parâmetro + correção de viés — simples de codar manualmente em `optimizer.rs`.

---

## 4. Por que 2 camadas ocultas com 16 neurônios?

- **Capacidade adequada ao problema:** a relação vazão → geração é uma função 1–2D suave; não precisa de uma rede profunda. 2×16 já é um aproximador universal suficiente.
- **Treino rápido em CPU:** poucos parâmetros (~ centenas de pesos) → cada época é barata, viável sem GPU num MacBook.
- **Menor risco de overfitting:** com dados relativamente limitados, uma rede pequena generaliza melhor e respeita melhor as restrições físicas.
- **Backprop manual tratável:** menos camadas = menos derivadas para implementar e validar no gradient check.

> Arquitetura: `entrada → [16, tanh] → [16, tanh] → saída (linear)`.

---

## 5. Como o resíduo físico é embutido na loss?

A PINN minimiza uma loss composta:

```
Loss_total = λ_d · Loss_dados + λ_f · Loss_fisica + λ_c · Loss_contorno
```

- **`Loss_dados`** — MSE entre geração prevista pela rede e geração real (ajuste aos dados observados).
- **`Loss_fisica`** — resíduo associado à **lei física de potência hidráulica**:

  ```
  P = ρ · g · Q · H · η
  ```

  onde `P` = potência (geração), `ρ` = densidade da água, `g` = gravidade, `Q` = vazão turbinada, `H` = queda líquida e `η` = eficiência. Como o projeto usa a vazão do posto Porto São José, que é uma proxy afluente a montante e não a vazão turbinada, a equação não é imposta como balanço direto. A forma usada é diferencial: `dP/dQ ≈ ρ·g·H·η` antes da saturação e `dP/dQ ≈ 0` após a saturação.

- **`Loss_contorno`** — condições físicas conhecidas:
  - geração = 0 quando vazão = 0;
  - geração ≤ capacidade instalada (~14 GW);
  - penalização de previsões acima da capacidade instalada.

Os pesos `λ_d, λ_f, λ_c` balanceiam ajuste aos dados vs. fidelidade física e são hiperparâmetros configuráveis; os valores finais estão na §10.

> Detalhe de implementação: as derivadas necessárias ao `Loss_fisica` são obtidas por **diferenças finitas** nas entradas — escolha registrada em `rust/src/neural/loss.rs`.

---

## 6. Como os datasets são cruzados (join por data)?

- **Granularidade:** geração é **horária**; vazão e ENA são **diárias**. Solução: agregar a geração para **diária** (média ou soma) em `limpar_dados.py`.
- **Chave do join:** a **data** (`YYYY-MM-DD`), após normalizar formatos de data de cada fonte.
- **Filtros antes do join (strings exatas confirmadas na Fase 1/2):**
  - ENA: `nom_bacia == "PARANA"` (match **exato** — cuidado: `PARANAIBA` e `PARANAPANEMA` são bacias distintas). Coluna usada: `ena_bruta_bacia_mwmed`.
  - vazão (complementar): `nom_postofluviometrico == "PORTO SAO JOSE"`, `nom_rio == "RIO PARANÁ"` (~5.400 m³/s). Coluna: `val_vazaomedia`.
- **Tabela cruzada:** `cruzar_datasets.py` produz `[data, geracao, ena_bruta, ena_armazenavel, vazao]` (ENA×geração via *inner*, vazão via *left*). O **núcleo do modelo** exportado é `[data, vazao, geracao]` — ver §8.
- **Período:** ENA + geração cobrem 2000–2024; o **modelo usa vazão → 2015–2024** (limite da fluviométrica). Ver §8.
- **Normalização:** feita em `exportar_para_rust.py` (min-max em [0, 1], ajustado só no treino 2015–2022), salvando os parâmetros em `normalizacao.json` para des-normalizar os resultados do Rust.

---

## 7. Métodos de referência para comparação

- A interpolação quadrática é mantida como baseline clássico, mas ela é fraca no regime de saturação porque a própria parábola tende a extrapolar e decair.
- A regressão linear com platô `P(Q)=min(a+bQ, Pmax)` foi adicionada como baseline saturante mais justo.
- Newton-Raphson permanece como método clássico de apoio para raízes/extremos 1D.
- A comparação revisada usa validação temporal: treino em 2015–2022 e teste em 2023–2024. A rede física-guiada 1D preserva uma curva saturante pós-processada, mas é limitada pela entrada única. `geracao_lag1` foi removida das entradas diretas; um MLP residual sobre persistência supera a persistência por margem pequena, enquanto modelos sem a âncora `P_{t-1}` ficam muito atrás.

---

## 8. Variável hidrológica de entrada: vazão proxy a montante

**Decisão (Fase 2, revisada após orientação):** usar a **vazão diária do posto `PORTO SAO JOSE`** (`RIO PARANÁ`, 2015–2024) como **proxy de disponibilidade hídrica a montante**, e tratar a **ENA da bacia do Paraná** (2000–2024) como **dado complementar explorado na EDA**.

> Variável de entrada = `vazao` (m³/s, média dos dois medidores do posto); alvo = `geracao` (MWmed). A vazão não é a vazão turbinada de Itaipu. Dataset do modelo: `[data, vazao, geracao]` (+ normalizados), período **2015–2024 (3.651 dias)**.

### Como chegamos aqui (a hipótese inicial foi revista pelos dados)

A hipótese inicial era usar a **ENA bruta** como entrada (mais 25 anos de dados; argumento de que a ENA já embute a produtividade da usina). A EDA da Fase 2 **contradisse** essa hipótese. Comparando na **mesma janela 2015–2024** (n=3.651) a correlação de cada variável com a geração diária de Itaipu:

| Variável | Pearson | Spearman |
|----------|:------:|:------:|
| **Vazão (PORTO SAO JOSE)** | **+0,587** | **+0,706** |
| ENA armazenável (bacia Paraná) | +0,476 | +0,495 |
| ENA bruta (bacia Paraná) | +0,457 | +0,494 |

Dois pontos decidiram a virada:

1. **A vazão local é mais preditiva** (Spearman 0,71 vs ~0,49). Ela representa melhor a disponibilidade hídrica próxima de Itaipu do que a ENA agregada da bacia inteira.
2. **A física favorece trabalhar com vazão, mas com ressalva.** Na equação `P = ρ·g·Q·H·η`, `Q` é a vazão turbinada. Como essa série não está nas bases locais usadas, Porto São José entra como proxy a montante; o termo físico atua como regularizador de inclinação, não como imposição direta do balanço hidráulico.

**Custo aceito:** período cai para **2015–2024** (~10 anos). Para uma PINN pequena (2×16), 3.651 pontos diários é suficiente.

### Nota sobre ENA bruta vs. armazenável (argumento preservado)
Na mesma janela, bruta e armazenável **empatam** em poder preditivo (Spearman 0,494 vs 0,495). Logo, se a ENA voltar a ser usada (ex.: modelo de 2 entradas no futuro), a escolha pela **bruta** se mantém pelo argumento de **não-vazamento de informação**: a armazenável já desconta vertimentos (decisão operacional que pertence ao lado da *saída*), enquanto a bruta = "quanto chegou".

> ⚠️ Correção da Fase 2: o posto `PORTO DOS PEREIRAS`, cogitado antes, fica no **`RIO PARANAÍBA`** (afluente, ~75 m³/s) — **não** serve como referência de Itaipu. O único posto relevante no Paraná principal é `PORTO SAO JOSE`, que tem **dois medidores** (id 64575000/64575001) cujas leituras diárias são **promediadas**.

**Como entra no artigo (texto-base para a Metodologia):**
> "Foram avaliadas como entrada a Energia Natural Afluente da bacia do Paraná (ENA, 2000–2024) e a vazão fluviométrica local no posto Porto São José, a montante de Itaipu (2015–2024). A análise exploratória mostrou correlação substancialmente maior da vazão local com a geração (Spearman 0,71 vs ~0,49 da ENA da bacia). Como essa vazão não é a vazão turbinada, ela é usada como proxy de disponibilidade hídrica; a física hidráulica entra como regularizador da inclinação e da saturação."

**Consequência prática:** o pipeline usa `[data, vazao, geracao]` como núcleo (2015–2024); a ENA permanece em `dataset_cruzado.csv` para as figuras exploratórias da EDA.

---

## 9. Validação da rede neural (Fase 4)

- **Gradient check:** diferença máxima entre gradiente analítico (backprop) e numérico (diferenças finitas) **< 1e-6**
  → backpropagation matematicamente correto.
- **Benchmark:** ajuste de `y = x²` com dados sintéticos
  → MSE final **0.000004**, redução de **65.510×** em relação ao início
  → `x=0.5` produz saída **0.2504** (erro < 0,2% vs. valor real 0.25).
- **Implementação:** Rust puro, **zero dependências de ML** (PRNG xorshift próprio para o init Xavier).
- **Otimizador Adam:** convergência estável, sem oscilação ou divergência.

> Estes números servem de baseline de sanidade: antes de adicionar a física (Fase 5), a rede já ajusta dados sintéticos corretamente. Qualquer regressão depois disso aponta para o termo `Loss_fisica`/contorno, não para o núcleo da rede.

---

## 10. Balanceamento da loss — PINN (Fase 5)

- **Pesos finais:** `λ = (λ_dados=1.0, λ_fisica=0.2, λ_contorno=1.0)`, escolhidos por validação interna e análise multissemente (2015–2020 treino, 2021–2022 validação; candidato precisa saturar para ser elegível).
- **Convergência da PINN física 1D (λ_fisica=0.2):**
  - `Loss_total`: 0.808 → 0.034.
  - `Loss_fisica`: 2.836 → 0.017 — resíduo físico reduzido.
  - `Loss_contorno`: ≈ 0.00069 — contorno satisfeito.
  - `loss_dados` estabiliza em ~0.030 (esperado — modela uma **curva de referência**, não o despacho diário).
- **Convergência da PINN multivariada (λ_fisica=0.2):**
  - `Loss_total`: 2.452 → 0.0174.
  - `Loss_dados`: 0.0154.
  - `Loss_fisica`: 0.0096.
- **Saturação da curva ajustada** em ~10.662 m³/s → `Pmax=13.946 MWmed`, definido como o máximo diário observado apenas no treino 2015–2022. Esse teto empírico é menor que a capacidade nominal de ~14 GW. O `q_sat` do termo físico é definido a priori por `Pmax/k`, não aprendido nesta versão por falta de vazão turbinada/variáveis operativas que tornem o parâmetro identificável.

### Seleção interna de `λ_fisica`

| `λ_fisica` | RMSE validação | Resíduo físico | Satura? |
|:---:|---:|---:|:---:|
| 0.0 | 1648 | 0.715 | ❌ |
| 0.05 | 1567 | 0.0856 | ✅ |
| 0.1 | 1559 | 0.0484 | ✅ |
| 0.2 | 1541 | 0.0290 | ✅ |
| 0.5 | 1539 | 0.0132 | ✅ |
| 1.0 | **1537** | 0.0170 | ✅ |

### Ablação de `λ_fisica` (executada — responde "o termo físico faz diferença?")

| `λ_fisica` | `loss_dados` | `loss_fisica` (resíduo) | Satura? | Interpretação |
|:------:|:-----------:|:--------------------:|:-------:|:--------------|
| **0.0** (sem resíduo físico) | **0.0238** ← menor | **0.559** ← enorme | ❌ | Melhor nos dados, **pior na física** |
| **0.2** | 0.0302 | 0.017 | ✅ ~10.662 m³/s | **Escolha final robusta** |
| **1.0** | 0.0325 | 0.0067 ← menor | ✅ ~10.662 m³/s | Menor resíduo físico, maior instabilidade |

> A linha λ=0 — "menor" e "enorme" na mesma linha — é o argumento visual: o leitor entende sem ler o texto.

**Leitura (resultado-chave do artigo — crítica metodológica ao MSE ingênuo):**
- **Sem o resíduo físico (λ_fisica=0, mantendo λ_contorno=1 na ablação): a curva NÃO satura** e o resíduo físico explode (0.559) — apesar de o `loss_dados` ser o **menor** dos três. Pior: a curva chega a **decair** em vazões altas, prevendo *menos* geração com *mais* água. Minimizar o MSE do termo de dados faz a rede rastrear a **média condicional** e falhar no comportamento de capacidade. **Menor loss de dados ≠ melhor modelo.**
- **λ_fisica alto (1.0):** física quase perfeita (resíduo 0.0067), mas `loss_dados` sobe e a sensibilidade a sementes aumenta — trade-off clássico de PINN.
- A varredura pontual escolheu `λ_fisica=1.0`, mas a análise com 10 sementes favoreceu `λ_fisica=0.2` (`1540±3` MWmed contra `1563±72` MWmed). Por isso, o modelo final usa `λ_fisica=0.2` e não reivindica superioridade geral de um peso específico.
- **Narrativa-base para o artigo (Resultados/Discussão):**
	  > "Na ablação, o modelo sem resíduo físico (λ_fisica=0, com contorno mantido) obteve o menor erro quadrático médio nos dados de treinamento (loss_dados=0,0238), porém falhou em reproduzir o comportamento de saturação, apresentando resíduo físico de 0,559. Este resultado demonstra empiricamente que minimizar o erro nos dados sem incorporar conhecimento físico produz um modelo estatisticamente ajustado porém fisicamente inválido — motivação central para a abordagem PINN."
- Dados: `data/outputs/ablacao_lambda.csv` (métricas) e `curvas_ablacao.csv` (3 curvas p/ a figura sobreposta).
- Seleção de λ: `data/outputs/selecao_lambda.csv`.

---

## 11. Validação temporal e análise retrospectiva — enquadramento honesto (Fase 6)

**Split executado:**
- Dataset supervisionado usado no Rust: 3.650 amostras após defasagens.
- Treino: 2015–2022, 2.919 dias.
- Teste temporal: 2023–2024, 731 dias.

**Métricas no teste temporal (2023–2024):**

| Modelo | RMSE teste | MAE teste | R² teste | Satura? |
|---|---:|---:|---:|:---:|
| Média do treino | 1994 | 1630 | -0.38 | n/a |
| Média mensal do treino | 1853 | 1540 | -0.19 | n/a |
| Interpolação quadrática | 1690 | 1339 | 0.01 | ❌ |
| Linear com platô | 1753 | 1433 | -0.07 | ✅ |
| Persistência (`P_t = P_{t-1}`) | 590 | 465 | 0.88 | n/a |
| Persistência + MLP residual (`α=0.30`) | 586 | 460 | 0.88 | n/a |
| MLP 1D | 1487 | 1213 | 0.23 | não imposta |
| PINN física 1D (λ=0.2) | 1776 | 1341 | -0.10 | ✅ |
| RF sem `P_{t-1}` | 1529 | 1192 | 0.19 | não imposta |
| MLP sem `P_{t-1}` | 1625 | 1235 | 0.08 | não imposta |
| Rede física sem `P_{t-1}` | 1672 | 1263 | 0.03 | ✅ |

**Decisão de enquadramento: manchete metodológica, SEM alegação de ganho financeiro.**
- A formulação deve ser descrita como rede neural guiada/regularizada por física, não como PINN clássica estrita.
- A PINN física 1D é válida como curva saturante pós-processada, mas não é o melhor preditor por RMSE no teste.
- O MLP 1D prevê melhor que a PINN física 1D, mas não impõe o requisito físico central (saturação).
- Sem `geracao_lag1` como entrada direta, os modelos hidrológicos/multivariados continuam limitados; isso reduz a força de uma interpretação causal hidráulica.
- A persistência é muito superior aos modelos sem âncora temporal; o residual sobre persistência mostra ganho pequeno. No split principal, o bootstrap pareado em blocos de 14 dias estimou redução de RMSE de 4,49 MWmed (IC95%: 1,05 a 7,74), mas a magnitude prática é baixa.
- A rede física-guiada multivariada melhora consistência física, mas não é o melhor preditor.
- A PINN física 1D superestima o teste: resíduo líquido +14,9 TWh = +9,9%.
- A rede física-guiada multivariada sem `P_{t-1}` superestima o teste em +14,8 TWh = +9,8%; o MLP sem `P_{t-1}` em +13,5 TWh = +9,0%; e a Random Forest sem `P_{t-1}` em +13,4 TWh = +8,9%.
- A vazão turbinada equivalente `Q_eq = P_real/k` foi calculada apenas como diagnóstico: no teste, Porto São José teve média 6361 m³/s, `Q_eq` média 8207 m³/s e correlação diária `r=0.54`.
- Validação móvel anual (2020–2024): RMSE médio persistência 596 MWmed; persistência + residual 593 MWmed; rede física-guiada sem `P_{t-1}` 1589 MWmed; MLP sem `P_{t-1}` 1584 MWmed; RF sem `P_{t-1}` 1754 MWmed.
- Benchmark complementar do MLP multivariado sem `P_{t-1}`: Rust release treinou 2.000 épocas em 8,55 s; Python puro, medido por 10 épocas e extrapolado, estimou ~562 s; NumPy vetorizado treinou em 1,71 s. Leitura correta: Rust é muito mais eficiente que Python interpretado em uma implementação do zero, mas não se reivindica superioridade genérica sobre bibliotecas numéricas otimizadas.
- **Causa física:** Itaipu tem **reservatório**, que desacopla a geração diária da vazão instantânea. Por isso, uma curva como função da vazão proxy diária deve ser interpretada como referência regularizada, não como ótimo operacional.
- **Resultado positivo defensável:** *"A validação temporal mostra que a geração diária é fortemente persistente; proxies públicas de vazão/ENA adicionam apenas ganho residual pequeno sobre a persistência, enquanto a rede regularizada por física explicita o trade-off entre erro preditivo e consistência hidráulica."*

**Trabalho futuro:** obter vazão turbinada, vertimento, nível do reservatório e queda líquida; redefinir o ótimo como **envelope superior** via regressão quantílica (perda pinball + física); e só aprender `q_sat` quando houver variáveis operativas suficientes para tornar o parâmetro identificável. Dados: `data/outputs/validacao_temporal.csv`, `predicoes_validacao_temporal.csv`, `diagnostico_vazao_turbinada_equivalente.csv`, `resumo_vazao_turbinada_equivalente.csv` e `simulacao_retrospectiva.csv`.

---

## 12. Exploração do platô temporal e residual

Nova rodada adicionada em `rust/src/experimentos.rs` para testar se o platô da persistência vinha de falta de memória temporal ou de falta de sinal nas entradas hidrológicas públicas.

**Regra de honestidade temporal:** as novas features usam somente `t-1` ou anterior. Foram testados três grupos:

- `hidro_lag1_basico`: sazonalidade + vazão/ENA em `t-1`.
- `hidro_lags_medias_30d`: lags `t-1, t-2, t-3, t-7, t-14, t-30`, médias móveis de 3/7/14/30 dias e variações de vazão/ENA.
- `operacional_lags_com_geracao`: hidrologia longa + geração passada (`P_{t-1}`, `P_{t-2}`, `P_{t-7}`, médias móveis e variações).

**Resultados no teste 2023–2024:**

| Modelo | Features | Formulação | RMSE teste | MAE teste | R² |
|---|---|---|---:|---:|---:|
| Persistência | n/a | `P_t=P_{t-1}` | 590.44 | 464.56 | 0.8789 |
| Persistência + delta mensal | n/a | residual mensal | 590.39 | 464.00 | 0.8789 |
| MLP residual | hidrologia longa | `P_t-P_{t-1}` | 578.77 | 452.66 | 0.8836 |
| Ridge residual | hidrologia longa | `P_t-P_{t-1}` | 580.10 | 454.35 | 0.8831 |
| MLP residual | operacional + geração passada | `P_t-P_{t-1}` | **507.55** | **390.86** | **0.9105** |
| MLP absoluto | operacional + geração passada | `P_t` | 509.06 | 395.61 | 0.9100 |

**Interpretação:**

- Mais memória hidrológica ajuda, mas pouco: o melhor residual hidrológico cai de 590.44 para 578.77 MWmed (~2,0%).
- A formulação residual é a mais informativa para responder se hidrologia explica a variação diária. A resposta é "sim, mas com sinal pequeno".
- A melhora grande aparece ao incluir geração passada como entrada operacional: 507.55 MWmed (~14,0% melhor que persistência). Isso é previsão operacional válida se `P_{t-1}` estiver disponível, mas o modelo deve ser apresentado como autorregressivo, não como evidência causal hidráulica.
- Prever anomalia mensal não resolveu o problema com hidrologia pura; modelos absolutos/anomalia ainda ficam muito acima da persistência quando não usam geração passada.
- A validação móvel anual específica do cenário operacional confirmou robustez: em 2020--2024, o MLP residual operacional venceu a persistência em todos os anos e reduziu o RMSE médio de 595.77 para 529.09 MWmed (~11,2% sobre as médias).

**Validação móvel operacional (média 2020--2024):**

| Modelo | RMSE médio | MAE médio | R² médio | Melhora vs. persistência |
|---|---:|---:|---:|---:|
| Persistência | 595.77 | 473.52 | 0.8372 | 0.00% |
| Ridge residual operacional | 535.29 | 421.73 | 0.8719 | 9.44% |
| RF residual operacional | 533.35 | 423.09 | 0.8729 | 10.10% |
| MLP residual operacional | **529.09** | **416.65** | **0.8754** | **11.19%** |
| MLP absoluto operacional | 569.59 | 449.90 | 0.8574 | 2.86% |

**Fator efetivo:** ajustar `P ≈ a + k_efetivo Q` com variáveis defasadas resultou em `k_efetivo` entre 0.567 e 0.779 MW/(m³/s), abaixo do `k_fisico≈1.042`. Isso reforça que Porto São José é proxy de disponibilidade hídrica, não vazão turbinada.

Dados e documentação completa: `docs/exploracao_plato_temporal_residual.md`, `data/outputs/exploracao_temporal_residual.csv`, `data/outputs/validacao_movel_operacional.csv`, `data/outputs/exploracao_fator_efetivo.csv`, `data/outputs/exploracao_features_temporais.csv`.

---

## Registro de decisões em aberto (a confirmar durante o desenvolvimento)

- [x] **Forma do termo `Loss_fisica`:** diferencial (`dP/dQ ≈ k`) via diferenças finitas — ver §8/§10.
- [x] **Valores finais de `λ_d, λ_f, λ_c`:** `(1.0, 0.2, 1.0)`, escolhidos por validação interna e sensibilidade multissemente (§10).
  - Trade-off confirmado empiricamente: λ_fisica=0 → não satura; λ_fisica=1.0 → melhor RMSE pontual, mas λ_fisica=0.2 → menor média e desvio-padrão em 10 sementes.
- [x] **Normalização:** min-max [0,1] ajustada somente no treino 2015–2022 (params em `normalizacao.json`).
- [x] **Validação temporal:** treino 2015–2022; teste 2023–2024 (§11).
- [x] **Conversão financeira:** **descartada** — a curva é referência regularizada, não ótimo operacional; enquadramento metodológico, sem alegar ganho em R$ (ver §11).
- [x] **Variável de entrada e janela:** **vazão** (posto PORTO SAO JOSE), **2015–2024** — decisão revista pela EDA, ver §8.
- [x] **Posto fluviométrico:** `PORTO SAO JOSE` (Rio Paraná), média dos 2 medidores. `PORTO DOS PEREIRAS` descartado (rio errado).
- [x] **ENA bruta vs. armazenável:** bruta (se a ENA voltar a ser usada) — empatam em correlação; bruta evita vazamento.
