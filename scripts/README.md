# Scripts Python

Python cuida só dos dados (download, limpeza, cruzamento e normalização), das figuras e das análises complementares. Todo o aprendizado acontece em Rust. O ponto de contato entre as duas partes é `data/processed/dataset_final.csv`.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, pandas e matplotlib com versões fixadas
```

Os scripts usam caminhos relativos à raiz do repositório e podem ser executados de qualquer pasta.

## Pré-processamento (fases 1–2)

| Ordem | Script | O que faz | Grava em |
|---|---|---|---|
| 1 | `download_datasets.py [ano_inicio ano_fim]` | baixa os três conjuntos do ONS pela API CKAN do portal; o snapshot versionado usa `2000 2024` | `data/raw/` |
| 2 | `limpar_dados.py` | filtra a bacia `PARANA` (match exato) e o posto `PORTO SAO JOSE`, promedia os dois medidores e agrega a geração horária em média diária | `data/processed/` |
| 3 | `cruzar_datasets.py` | junta geração, ENA e vazão por data | `data/processed/dataset_cruzado.csv` |
| 4 | `exportar_para_rust.py` | cria defasagens e sazonalidade, ajusta a normalização min-max só no treino 2015–2022 e grava a entrada do Rust | `data/processed/dataset_final.csv`, `normalizacao.json` |

O passo 1 é opcional: o snapshot de 03/06/2026 já está em `data/raw/`, e os passos 2 a 4 o transformam em `data/processed/` byte a byte.

## Análises complementares

| Script | O que faz | Grava em |
|---|---|---|
| `verificar_reprodutibilidade.py [--ref HEAD] [-v]` | compara `data/processed/` e `data/outputs/` com a versão commitada, ignorando só as colunas de tempo (`tempo_*`); usa apenas a biblioteca padrão | — (sai com código 1 se algo não bate) |
| `teste_bootstrap_residual_persistencia.py` | bootstrap pareado em blocos (7, 14 e 30 dias; 20.000 reamostragens; semente fixa) comparando persistência e modelos residuais | `data/outputs/bootstrap_residual_persistencia.csv` |
| `benchmark_python.py [--pure-epochs N] [--numpy-epochs N]` | treina o mesmo MLP em Python puro e em NumPy e compara com o tempo do Rust | `data/outputs/benchmark_python.csv`, `benchmark_comparativo.csv` |
| `gerar_graficos.py` | figuras do artigo e tabela comparativa de modelos | `artigo/figuras/` (local, fora do git), `data/outputs/comparativo_modelos.csv` |
| `gerar_graficos_en.py` | as mesmas figuras com rótulos em inglês | `artigo/figuras_en/` (local, fora do git) |

Os tempos de execução dependem da máquina, então os CSVs de benchmark mudam a cada execução. Os demais resultados são determinísticos.
