# Dados brutos (`data/raw/`)

Snapshot dos três conjuntos públicos do **Operador Nacional do Sistema Elétrico (ONS)** usados no projeto, baixados do [Portal de Dados Abertos do ONS](https://dados.ons.org.br) em **3 de junho de 2026** e versionados sem nenhuma alteração.

- **Licença:** Creative Commons Atribuição (CC-BY), conforme o portal do ONS. Fonte: ONS — Portal de Dados Abertos.
- **Integridade:** [`SHA256SUMS`](SHA256SUMS) traz a soma SHA-256 de cada arquivo. Para conferir: `sha256sum -c SHA256SUMS` (Linux) ou `shasum -a 256 -c SHA256SUMS` (macOS). A CI repete essa verificação a cada push.
- **Novo download:** `python scripts/python/download_datasets.py 2000 2024` descobre os arquivos pela API CKAN do portal e os grava aqui. O ONS pode revisar séries antigas, então um download novo pode diferir do snapshot; as somas mostram exatamente quais arquivos mudaram.

## Cobertura do snapshot

| Conjunto | Arquivos | Período no snapshot | Granularidade |
|---|---|---|---|
| Geração de Itaipu | `GERACAO_ITAIPU.csv` | 01/01/2000 a 02/06/2026 | horária |
| Grandezas fluviométricas | `GRANDEZAS_FLUVIOMETRICAS_2015.csv` … `_2024.csv` | 2015 a 2024 (a série começa em 2015) | diária |
| ENA por bacia | `ENA_DIARIO_BACIAS_2000.csv` … `_2024.csv` | 2000 a 2024 | diária |

O artigo usa os registros até 31/12/2024: de 2000 a 2024 para geração e ENA e de 2015 a 2024 para a vazão. O modelo usa a janela comum de 01/01/2015 a 31/12/2024. Todos os arquivos usam `;` como separador e codificação UTF-8.

## Geração de Itaipu

- **Portal:** https://dados.ons.org.br/dataset/geracao_itaipu
- **Conteúdo:** 231.440 registros horários, sem valores ausentes; máximo de 14.348,9 MW (20/06/2016, 10h).
- **Variável-alvo:** `val_itaipu_total`, agregada em média diária (MWmed) por `limpar_dados.py`.

| Coluna | Descrição | Unidade |
|---|---|---|
| `din_instante` | data e hora da medição | timestamp |
| `val_itaipu_total` | geração total de Itaipu | MW |
| `val_itaipu_60hz` | geração no lado 60 Hz (Brasil) | MW |
| `val_itaipu_50hz` | geração no lado 50 Hz (Paraguai) | MW |
| `val_itaipu_50hz_br` | parcela de 50 Hz convertida para o Brasil | MW |
| `val_itaipu_br` | parcela total destinada ao Brasil | MW |
| `val_itaipu_py` | parcela destinada ao Paraguai | MW |

## Grandezas fluviométricas

- **Portal:** https://dados.ons.org.br/dataset/grandezas_fluviometricas
- **Conteúdo:** todos os postos fluviométricos do país, um arquivo por ano.
- **Posto usado:** `PORTO SAO JOSE`, no `RIO PARANÁ`, a montante de Itaipu. O posto tem dois medidores (ids 64575000 e 64575001), cujas leituras diárias são promediadas. O posto `PORTO DOS PEREIRAS`, cogitado no início, foi descartado por ficar no Rio Paranaíba, um afluente.

| Coluna | Descrição | Unidade |
|---|---|---|
| `id_postofluv` | código do posto | inteiro |
| `nom_postofluviometrico` | nome do posto (chave do filtro) | texto |
| `val_latitude`, `val_longitude` | localização | graus |
| `nom_rio` | rio | texto |
| `nom_bacia` | bacia | texto |
| `din_medicao` | data da medição | data |
| `val_vazaomedia` | vazão média diária | m³/s |
| `val_vazaomediaincr` | vazão média incremental | m³/s |

## Energia Natural Afluente (ENA) por bacia

- **Portal:** https://dados.ons.org.br/dataset/ena-diario-por-bacia
- **Bacia usada:** `PARANA`, com filtro de igualdade exata, porque `PARANAIBA` e `PARANAPANEMA` são bacias distintas com o mesmo prefixo.
- **Variável:** `ena_bruta_bacia_mwmed`. A ENA bruta é preferida à armazenável porque a armazenável já desconta vertimentos, uma decisão operacional.

| Coluna | Descrição | Unidade |
|---|---|---|
| `nom_bacia` | nome da bacia (filtro: `PARANA`) | texto |
| `ena_data` | data | data |
| `ena_bruta_bacia_mwmed` | ENA bruta da bacia | MWmed |
| `ena_bruta_bacia_percentualmlt` | ENA bruta em % da média de longo termo | % |
| `ena_armazenavel_bacia_mwmed` | ENA armazenável | MWmed |
| `ena_armazenavel_bacia_percentualmlt` | ENA armazenável em % da média de longo termo | % |
