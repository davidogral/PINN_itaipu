# 📚 Glossário

Termos técnicos do projeto explicados de forma simples. 

---

### PINN (Physics-Informed Neural Network)
Rede neural que aprende **respeitando leis da física**, geralmente por resíduos de equações diferenciais avaliados na função de perda. No artigo, o termo é tratado com cuidado: como usamos vazão proxy a montante, e não vazão turbinada, a formulação atual é melhor descrita como rede neural guiada/regularizada por física, inspirada em PINNs.

### Rede neural guiada por física
Rede neural treinada com conhecimento físico adicional na perda, nas entradas, na arquitetura ou no pós-processamento. Neste projeto, a física entra como regularização da inclinação hidráulica esperada e do regime de capacidade.

### Rede neural feedforward
Tipo mais simples de rede: a informação flui só "para frente", da entrada para a saída, passando por camadas que multiplicam por pesos, somam vieses e aplicam uma função de ativação.

### Camada oculta
Camada de neurônios entre a entrada e a saída. "Oculta" porque não vemos seus valores diretamente — são representações internas que a rede aprende.

### Neurônio
Unidade que recebe vários números, faz uma soma ponderada (pesos + viés) e aplica uma ativação.

### Peso (weight) e viés (bias)
**Peso**: o quanto cada entrada influencia o neurônio. **Viés**: um deslocamento constante somado antes da ativação. São os parâmetros que o treino ajusta.

### Função de ativação
Função não linear aplicada na saída do neurônio. Sem ela, a rede seria apenas uma combinação linear. Aqui usamos **tanh**.

### tanh (tangente hiperbólica)
Função suave que mapeia qualquer número para o intervalo (−1, 1). Sua derivada é fácil: `1 − tanh²(x)`. Boa para perdas físicas por ser infinitamente diferenciável.

### Forward pass
O cálculo "para frente": dada uma entrada, percorrer as camadas até produzir a saída (a previsão da rede).

### Backpropagation (retropropagação)
Algoritmo que calcula **como cada peso contribuiu para o erro**, propagando o erro da saída de volta para as entradas usando a regra da cadeia. Fornece os gradientes para o otimizador.

### Gradiente
Vetor que aponta a direção de maior crescimento da função de perda. O treino caminha no sentido **oposto** ao gradiente para reduzir o erro.

### Gradient descent (descida do gradiente)
Estratégia de otimização: ajustar os pesos um pouco na direção que diminui a perda, repetidamente.

### Adam
Otimizador que melhora a descida do gradiente usando **momentum** (memória da direção anterior) e **escala adaptativa** por parâmetro. Converge rápido e é estável.

### Função de perda (loss)
Número que mede o quão errada está a rede. Treinar = minimizar a loss. Na rede física-guiada, a loss soma três partes: dados + resíduo físico + contorno.

### Loss_dados
Parte da perda que mede o erro entre a previsão e os dados reais (geralmente **MSE**).

### Loss_física
Parte da perda que mede o quanto a rede **viola a regularização física**. Neste projeto, ela penaliza diferenças entre a inclinação da curva aprendida e a inclinação esperada pela lei de potência hidráulica no regime produtivo. Não há uma EDP completa do reservatório sendo resolvida.

### Loss_contorno
Parte da perda que força a rede a respeitar **condições de fronteira** conhecidas (ex.: geração = 0 quando vazão = 0).

### MSE (Mean Squared Error)
Erro quadrático médio: média dos quadrados das diferenças entre previsto e real.

### EDP / EDO
**Equação Diferencial Parcial / Ordinária**: equação que relaciona uma grandeza com suas taxas de variação (derivadas). Descreve leis físicas.

### Ponto de colocação (collocation point)
Pontos do domínio onde o modelo avalia o resíduo físico durante o treino. Neste projeto, os pontos físicos coincidem com as amostras do treino em full-batch.

### Interpolação quadrática
Técnica que ajusta uma **parábola** passando por 3 pontos conhecidos para estimar valores intermediários. Implementada por Lagrange ou Newton.

### Newton-Raphson
Método iterativo clássico para achar raízes (ou extremos, via derivada) de uma função, usando a tangente para se aproximar da solução passo a passo.

### Vazão (Q)
Volume de água que passa por um ponto do rio por segundo. Medida em **m³/s**. No projeto, a entrada é a vazão do posto Porto São José, usada como **proxy de disponibilidade hídrica a montante**, não como vazão turbinada de Itaipu.

### ENA (Energia Natural Afluente)
Estimativa do potencial energético associado à água afluente de uma bacia. Medida em **MWmed**. No projeto, é usada na PINN multivariada como covariável hidrológica complementar.

### Variáveis defasadas
Valores observados em dias anteriores, como `vazao_lag1` e `geracao_lag1`. Ajudam a medir persistência temporal. Na versão revisada, `vazao_lag1` pode entrar nos modelos treináveis, mas `geracao_lag1` fica apenas no baseline de persistência para evitar que a rede aprenda a repetir a geração do dia anterior.

### Queda líquida (H)
Diferença de altura efetiva que a água percorre na turbina, descontadas as perdas. Em metros.

### Produtividade relativa
Quanta geração aparece por unidade da vazão proxy usada no modelo (ex.: MWmed por m³/s). É uma métrica descritiva da curva de referência, não uma prova direta de eficiência operacional da usina.

### Vazão turbinada equivalente
Vazão que seria compatível com a geração observada pela equação `Q_eq = P_real / (rho*g*H*eta)`. Como é calculada a partir da própria geração real, serve apenas como diagnóstico físico e não pode ser usada como entrada preditiva sem vazar o alvo.

### Modelo residual sobre persistência
Modelo que parte da previsão ingênua `P(t-1)` e aprende só uma correção pequena com variáveis hidrológicas: `P_hat(t) = P(t-1) + alpha*f(x_t)`. No projeto, `geracao_lag1` não entra em `f(x_t)`; ela aparece apenas na âncora explícita de persistência.

### Capacidade instalada
Potência máxima que a usina consegue gerar. Em Itaipu, ~14 GW. Limite físico usado como condição de contorno.

### Simulação retrospectiva
Reaplicar o modelo sobre o **passado real**: comparar a geração observada com a curva de referência. Na versão atual, o modelo é treinado em 2015–2022 e avaliado também em 2023–2024, então a retrospectiva separa resíduos de treino e de teste temporal. Mesmo assim, não serve para afirmar ganho energético ou financeiro.

### Validação temporal
Separação cronológica dos dados: o modelo aprende em um período antigo e é avaliado em um período posterior não visto. Neste projeto: treino em 2015–2022 e teste em 2023–2024.

### Validação interna
Divisão usada para escolher hiperparâmetros sem tocar no teste final. Neste projeto, `λ_fisica` é escolhido treinando em 2015–2020 e validando em 2021–2022; o teste 2023–2024 fica preservado para avaliação final.

### ONS
Operador Nacional do Sistema Elétrico — fornece os dados públicos usados no projeto.

### Normalização
Reescalar os dados (ex.: para [0,1] ou média 0 / desvio 1) para o treino ficar estável. Os parâmetros são guardados para depois "des-normalizar".

### Gradient check
Teste que compara o gradiente do backprop com uma estimativa por **diferenças finitas**, para verificar se a retropropagação está correta.

### MWh / MW
**MW** (megawatt) = potência (taxa instantânea). **MWh** (megawatt-hora) = energia (potência × tempo). 1 MW por 1 hora = 1 MWh.
