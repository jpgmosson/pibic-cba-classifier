"""Gera municipios_sc.csv a partir de 05_banco_disc_mineracao_final.xlsx.

O .xlsx foi fornecido pelo professor Fábio (disciplina de mineração de dados) e NÃO
é versionado; só o CSV limpo entra no repositório. Requer openpyxl (usado só aqui,
não faz parte do requirements.txt). Uso:

    python datasets/preparar_municipios_sc.py

Aba "banco": as linhas 1-18 são um painel de estatísticas pré-calculado (não são
dados) e a tabela real começa na linha 20 (cabeçalho), com 295 municípios e 42
colunas. Tratamento (justificativas em docs/ADENDO_Municipios_SC.md):

1. identificadores removidos: cities, Codigo_IBGE;
2. colunas que determinam ou aproximam o alvo removidas (EXCLUIR_ALVO);
3. gap_medio_pd (57% ausente) e total_instrumentos (17% ausente) removidas;
4. linhas com ausente em growth_2000_2010, pop_2010_2022, growth_%_2010_2022,
   IDH_2010 ou predominance_2010 descartadas (3 linhas, todas Urbano).

Nenhum valor é imputado nem alterado nas colunas mantidas.
"""

from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
XLSX = here / "05_banco_disc_mineracao_final.xlsx"
SAIDA = here / "municipios_sc.csv"

ALVO = "predominance_2022"
IDENTIFICADORES = ["cities", "Codigo_IBGE"]
# Pop_urb/Pop_rur/Domicilios_*/dens_*: determinam o alvo (alvo == Pop_urb/Population_2022 >= 0,5).
# area_urb/area_rur/Urb %/Area_Pred: proxies da mesma classificação censitária urbano/rural.
# change_2010_2022: depende diretamente do alvo.
EXCLUIR_ALVO = [
    "Pop_urb", "Pop_rur", "Domicilios_urb", "Domicilios_rur", "area_urb", "area_rur",
    "dens_urb_22", "dens_rur", "Urb %", "Area_Pred", "change_2010_2022",
]  # fmt: skip
EXCLUIR_AUSENTES = ["gap_medio_pd", "total_instrumentos"]
DESCARTAR_LINHA_SE_AUSENTE = [
    "growth_2000_2010", "pop_2010_2022", "growth_%_2010_2022", "IDH_2010", "predominance_2010",
]  # fmt: skip

df = pd.read_excel(XLSX, sheet_name="banco", header=19)
assert df.shape == (295, 42) and df["cities"].is_unique
assert df[ALVO].notna().all() and df[ALVO].value_counts().to_dict() == {"Urbano": 209, "Rural": 86}
# Definição do alvo (justifica EXCLUIR_ALVO): urbano quando Pop_urb / Population_2022 >= 0,5.
assert ((df["Pop_urb"] / df["Population_2022"] >= 0.5).map({True: "Urbano", False: "Rural"}) == df[ALVO]).all()

df = df.drop(columns=IDENTIFICADORES + EXCLUIR_ALVO + EXCLUIR_AUSENTES)
df = df.dropna(subset=DESCARTAR_LINHA_SE_AUSENTE)
assert len(df) == 292 and df.notna().all().all() and df.shape[1] == 27


def _texto(valor):
    """year_MP mistura anos (int) com 'non_located': tudo vira texto, sem '.0'."""
    return str(int(valor)) if isinstance(valor, float) and valor.is_integer() else str(valor)


df["year_MP"] = df["year_MP"].map(_texto)
# O alvo vai por último, como no german_credit.csv.
df = df[[c for c in df.columns if c != ALVO] + [ALVO]]
df.to_csv(SAIDA, index=False, encoding="utf-8", lineterminator="\n")
