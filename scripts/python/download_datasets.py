#!/usr/bin/env python3
"""
Baixa os 3 datasets públicos do ONS para data/raw/, usando apenas a
biblioteca padrão do Python (urllib). 

Datasets (estrutura confirmada via API CKAN do portal dados.ons.org.br):

  1. Geração de Itaipu — base HORÁRIA, arquivo único 
       https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_itaipu/GERACAO_ITAIPU.csv
       Colunas: din_instante; val_itaipu_total; val_itaipu_60hz;
                val_itaipu_50hz; val_itaipu_50hz_br; val_itaipu_br; val_itaipu_py

  2. Grandezas Fluviométricas — base DIÁRIA, um CSV POR ANO
       dataset CKAN: grandezas_fluviometricas
       Colunas: id_postofluv; nom_postofluviometrico; val_latitude;
                val_longitude; nom_rio; nom_bacia; din_medicao;
                val_vazaomedia; val_vazaomediaincr

  3. ENA Diário por Bacia — base DIÁRIA, um CSV POR ANO 
       dataset CKAN: ena-diario-por-bacia
       Colunas: nom_bacia; ena_data; ena_bruta_bacia_mwmed;
                ena_bruta_bacia_percentualmlt; ena_armazenavel_bacia_mwmed;
                ena_armazenavel_bacia_percentualmlt
       (a bacia relevante para Itaipu é "PARANA")

Uso:
    python scripts/python/download_datasets.py            # janela padrão
    python scripts/python/download_datasets.py 2010 2024  # ano_inicio ano_fim

Todos os CSVs do ONS usam separador ';'.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

# Configuração
# Janela temporal padrão (CSVs anuais de vazão/ENA). Pode ser sobrescrita por
# argumentos de linha de comando. A geração horária é arquivo único (toda a série).
ANO_INICIO_PADRAO = 2020
ANO_FIM_PADRAO = 2024

PROJ_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJ_ROOT / "data" / "raw"

CKAN_PACKAGE = "https://dados.ons.org.br/api/3/action/package_show?id={}"

GERACAO_URL = (
    "https://ons-aws-prod-opendata.s3.amazonaws.com/"
    "dataset/geracao_itaipu/GERACAO_ITAIPU.csv"
)

# dataset_id no CKAN  ->  rótulo amigável para os logs
DATASETS_ANUAIS = {
    "grandezas_fluviometricas": "Grandezas Fluviométricas",
    "ena-diario-por-bacia": "ENA Diário por Bacia",
}

TIMEOUT = 120
_ANO_RE = re.compile(r"_(\d{4})\.csv$", re.IGNORECASE)


# Utilidades 
def tamanho_legivel(n: int | None) -> str:
    if not n:
        return "?"
    for unidade in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unidade}"
        n /= 1024
    return f"{n:.1f} TB"


def tamanho_remoto(url: str) -> int | None:
    """Retorna o Content-Length do recurso (via HEAD), ou None."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def baixar(url: str, destino: Path) -> bool:
    """Baixa `url` para `destino` com barra de progresso.

    Pula o download se o arquivo já existir com o mesmo tamanho do remoto.
    Retorna True se o arquivo está disponível ao final.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    remoto = tamanho_remoto(url)

    if destino.exists() and remoto is not None and destino.stat().st_size == remoto:
        print(f"   ✓ já existe ({tamanho_legivel(remoto)}): {destino.name}")
        return True

    print(f"   ↓ {destino.name}  ({tamanho_legivel(remoto)})")
    tmp = destino.with_suffix(destino.suffix + ".part")

    def _progresso(blocos: int, tam_bloco: int, total: int) -> None:
        if total <= 0:
            return
        baixado = min(blocos * tam_bloco, total)
        pct = 100 * baixado / total
        barra = "#" * int(pct // 4)
        sys.stdout.write(f"\r     [{barra:<25}] {pct:5.1f}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progresso)
        sys.stdout.write("\n")
        tmp.replace(destino)
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write("\n")
        print(f"     ✗ FALHA: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def recursos_csv(dataset_id: str) -> list[dict]:
    """Consulta a API CKAN e devolve os recursos CSV do dataset."""
    url = CKAN_PACKAGE.format(dataset_id)
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        dados = json.load(resp)
    recursos = dados["result"]["resources"]
    return [r for r in recursos if (r.get("format") or "").upper() == "CSV"]


def ano_do_recurso(url: str) -> int | None:
    m = _ANO_RE.search(url)
    return int(m.group(1)) if m else None


# Fluxo principal 
def main(ano_inicio: int, ano_fim: int) -> int:
    print("=" * 64)
    print("  PGNN-Itaipu — Fase 1: download dos datasets (ONS)")
    print(f"  Janela anual (vazão/ENA): {ano_inicio}–{ano_fim}")
    print(f"  Destino: {RAW_DIR}")
    print("=" * 64)

    falhas: list[str] = []

    # 1) Geração de Itaipu (arquivo único) 
    print("\n[1/3] Geração de Itaipu (horária, arquivo único)")
    if not baixar(GERACAO_URL, RAW_DIR / "GERACAO_ITAIPU.csv"):
        falhas.append("Geração de Itaipu")

    # 2) e 3) Datasets anuais (descoberta via CKAN) 
    for i, (dataset_id, rotulo) in enumerate(DATASETS_ANUAIS.items(), start=2):
        print(f"\n[{i}/3] {rotulo} (diária, um CSV por ano)")
        try:
            recursos = recursos_csv(dataset_id)
        except Exception as exc:  # noqa: BLE001
            print(f"   ✗ não foi possível consultar a API CKAN: {exc}")
            falhas.append(rotulo)
            continue

        alvos = []
        for r in recursos:
            ano = ano_do_recurso(r["url"])
            if ano is not None and ano_inicio <= ano <= ano_fim:
                alvos.append((ano, r["url"]))
        alvos.sort()

        if not alvos:
            print(f"   ✗ nenhum CSV anual encontrado em {ano_inicio}–{ano_fim}")
            falhas.append(rotulo)
            continue

        print(f"   {len(alvos)} arquivo(s) anual(is) na janela")
        for ano, url in alvos:
            nome = url.rsplit("/", 1)[-1]
            if not baixar(url, RAW_DIR / nome):
                falhas.append(f"{rotulo} {ano}")

    # Resumo 
    print("\n" + "=" * 64)
    if falhas:
        print("  Concluído COM FALHAS em:")
        for f in falhas:
            print(f"    - {f}")
        print("  (ver instruções manuais em data/raw/README.md)")
        return 1
    print("  ✓ Todos os datasets foram baixados para data/raw/")
    print("  Próximo passo: notebooks 01/02 (EDA) e depois limpar_dados.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    inicio, fim = ANO_INICIO_PADRAO, ANO_FIM_PADRAO
    if len(sys.argv) == 3:
        inicio, fim = int(sys.argv[1]), int(sys.argv[2])
    elif len(sys.argv) != 1:
        print("uso: python download_datasets.py [ano_inicio ano_fim]")
        raise SystemExit(2)
    raise SystemExit(main(inicio, fim))
