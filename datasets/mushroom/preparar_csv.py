"""Gera mushroom.csv (com cabeçalho) a partir do arquivo bruto agaricus-lepiota.data.

O arquivo bruto da UCI não tem cabeçalho; os nomes abaixo seguem a seção
"Attribute Information" de agaricus-lepiota.names (o "?" de "bruises?" foi
removido do nome da coluna). Nenhum valor é alterado: o "?" de stalk-root
é mantido como categoria (ver README.md). Uso:

    python datasets/mushroom/preparar_csv.py
"""

from pathlib import Path

COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]  # fmt: skip

here = Path(__file__).resolve().parent
rows = (here / "agaricus-lepiota.data").read_text(encoding="utf-8").split()
assert len(rows) == 8124 and all(len(r.split(",")) == len(COLUMNS) for r in rows)
(here / "mushroom.csv").write_text(",".join(COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
