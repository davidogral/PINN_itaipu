# Referências de Leitura

Materiais a ler **antes de implementar** cada parte do projeto. Estão em ordem sugerida de leitura, do fundamento ao aprofundamento.

---

## 1. Paper original de PINNs — Raissi, Perdikaris & Karniadakis (2019)

- **Título:** *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations*
- **Publicado em:** *Journal of Computational Physics*, vol. 378, 2019.
- **Link:** https://doi.org/10.1016/j.jcp.2018.10.045
- **Por que ler:** É a fonte primária do conceito. Define como embutir uma EDP na função de perda e por que `tanh` é usada. Leitura **obrigatória** antes da Fase 5 (PINN). Foque nas seções de formulação da loss e nos pontos de colocação.

---

## 2. Physics-Informed Machine Learning — Karniadakis et al. (2021)

- **Título:** *Physics-Informed Machine Learning*
- **Publicado em:** *Nature Reviews Physics*, vol. 3, 2021.
- **Link:** https://doi.org/10.1038/s42254-021-00314-5
- **Por que ler:** Ajuda a diferenciar PINNs clássicas de métodos mais amplos de aprendizado de máquina guiado por física.

---

## 3. Physics-Guided Neural Networks — Daw et al. (capítulo, 2022)

- **Título:** *Physics-Guided Neural Networks (PGNN): An Application in Lake Temperature Modeling*
- **Publicado em:** *Knowledge-Guided Machine Learning*, Chapman and Hall/CRC, pp. 353--372, 2022.
- **Link:** https://doi.org/10.1201/9781003143376-15
- **Por que ler:** Referência útil para justificar a nomenclatura de rede guiada por física quando a física entra como regularização, e não como solução direta de uma EDP.
- **Observação:** existe versão preprint em arXiv:1710.11431.

---

## 4. Introdução a redes neurais para físicos — de Lima, Miranda & Farias (2026)

- **Título:** *Introdução às Redes Neurais para Físicos*
- **Publicado em:** *Revista Brasileira de Ensino de Física*, vol. 48, e20250183, 2026.
- **Link:** https://doi.org/10.1590/1806-9126-rbef-2025-0183
- **Por que ler:** Conecta a matemática de redes neurais (forward, backprop, gradientes) com a intuição física. Útil antes da Fase 4 (implementar a rede do zero), pois trata as derivadas de forma explícita.

---

## 5. Review de PINNs — Springer Nature (2022)

- **Tipo:** Artigo de revisão (survey) sobre PINNs.
- **Publicado em:** *Journal of Scientific Computing*, 2022.
- **Link:** https://doi.org/10.1007/s10915-022-01939-z
- **Por que ler:** Panorama do estado da arte — variações de arquitetura, estratégias de balanceamento da loss (pesos `λ`), aplicações e limitações conhecidas. Útil para as seções de **trabalhos relacionados** e **discussão** do artigo.

---

## Leituras complementares (opcionais)

- **Deep Learning (Goodfellow, Bengio & Courville, 2016)** — livro-base para redes neurais, MLPs, backpropagation e otimização. Link: https://www.deeplearningbook.org/
- **Adam (Kingma & Ba, 2015)** — *Adam: A Method for Stochastic Optimization*, trabalho apresentado na ICLR 2015. Link: https://mlanthology.org/iclr/2015/kingma2015iclr-adam/
- **Digitalização e previsão em hidrelétricas (Pepe, 2024)** — revisão sobre dados, KPIs, modelagem e previsão em usinas hidrelétricas. Link: https://doi.org/10.3390/en17040941
- **Previsão hidrológica com deep learning (Zhao et al., 2024)** — revisão sobre métodos de aprendizado profundo para previsão hidrológica. Link: https://doi.org/10.3390/w16101407
- **Documentação ONS Open Data** — páginas oficiais dos conjuntos de dados usados no artigo:
  - Geração de Itaipu: https://dados.ons.org.br/dataset/geracao_itaipu
  - Grandezas fluviométricas: https://dados.ons.org.br/dataset/grandezas_fluviometricas
  - ENA diário por bacia: https://dados.ons.org.br/dataset/ena-diario-por-bacia
- **Análise Numérica (qualquer livro-texto)** — capítulos de interpolação de Lagrange/Newton e do método de Newton-Raphson (Fase 3).

---
