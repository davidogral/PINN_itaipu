#!/usr/bin/env python3
"""Confere se os dados regenerados reproduzem os arquivos versionados no git.

Compara cada arquivo de data/processed/ e data/outputs/ com a versão do commit
indicado (padrão: HEAD). CSVs são comparados célula a célula, ignorando as
colunas cujo nome começa com "tempo": elas guardam tempos de execução, que
variam entre máquinas e execuções. Os demais arquivos são comparados byte a byte.

Uso, na raiz do repositório, depois de rodar o pipeline:
    python scripts/python/verificar_reprodutibilidade.py [--ref HEAD] [-v]

Sai com código 0 se tudo foi reproduzido e 1 caso contrário. Usa apenas a
biblioteca padrão.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
PASTAS = ("data/processed", "data/outputs")


def arquivos_versionados(ref: str) -> list[str]:
    saida = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", *PASTAS],
        cwd=RAIZ,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [linha for linha in saida.splitlines() if not linha.endswith(".gitkeep")]


def conteudo_versionado(ref: str, caminho: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{ref}:{caminho}"], cwd=RAIZ, check=True, capture_output=True
    ).stdout


def diferenca_csv(esperado: bytes, obtido: bytes) -> str | None:
    """Devolve a primeira diferença fora das colunas de tempo, ou None."""
    linhas_esp = list(csv.reader(io.StringIO(esperado.decode("utf-8"))))
    linhas_obt = list(csv.reader(io.StringIO(obtido.decode("utf-8"))))
    if not linhas_esp or not linhas_obt or linhas_esp[0] != linhas_obt[0]:
        return "cabeçalho diferente"
    if len(linhas_esp) != len(linhas_obt):
        return f"{len(linhas_obt) - 1} linhas (esperado: {len(linhas_esp) - 1})"
    cabecalho = linhas_esp[0]
    colunas = [j for j, nome in enumerate(cabecalho) if not nome.startswith("tempo")]
    for i, (esp, obt) in enumerate(zip(linhas_esp[1:], linhas_obt[1:]), start=2):
        if len(esp) != len(obt):
            return f"linha {i}: número de colunas diferente"
        for j in colunas:
            if esp[j] != obt[j]:
                return f"linha {i}, coluna {cabecalho[j]}: {obt[j]} (esperado: {esp[j]})"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD", help="commit de referência (padrão: HEAD)")
    parser.add_argument("-v", "--verbose", action="store_true", help="lista também os arquivos reproduzidos")
    args = parser.parse_args()

    arquivos = arquivos_versionados(args.ref)
    falhas = 0
    for caminho in arquivos:
        local = RAIZ / caminho
        if not local.exists():
            print(f"FALTA      {caminho}")
            falhas += 1
            continue
        esperado = conteudo_versionado(args.ref, caminho)
        obtido = local.read_bytes()
        if caminho.endswith(".csv"):
            problema = diferenca_csv(esperado, obtido)
        else:
            problema = None if esperado == obtido else "conteúdo diferente"
        if problema:
            print(f"DIFERENTE  {caminho}: {problema}")
            falhas += 1
        elif args.verbose:
            print(f"ok         {caminho}")

    total = len(arquivos)
    print(f"{total - falhas} de {total} arquivos reproduzidos (colunas de tempo ignoradas).")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
