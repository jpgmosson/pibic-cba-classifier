"""Gera german_credit.csv (com cabeçalho) a partir do arquivo bruto german.data.

O arquivo bruto da UCI é separado por espaço, sem cabeçalho, com a classe na
21ª coluna (1 = bom pagador, 2 = mau pagador). Os nomes abaixo seguem a seção
"Attribute description for german" de german.doc. Nenhum valor é alterado
(os códigos A11, A12, ... e a classe 1/2 são mantidos). Uso:

    python datasets/german_credit/preparar_csv.py
"""

from pathlib import Path

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk",
]  # fmt: skip

here = Path(__file__).resolve().parent
rows = [line.split() for line in (here / "german.data").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 1000 and all(len(r) == len(COLUMNS) for r in rows)
lines = [",".join(COLUMNS)] + [",".join(r) for r in rows]
(here / "german_credit.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
