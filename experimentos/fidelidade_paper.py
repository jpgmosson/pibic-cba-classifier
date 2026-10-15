"""Aproxima a baseline das condições do artigo do CBA (Liu, Hsu e Ma, 1998) e compara.

    python -m experimentos.fidelidade_paper

Referência do artigo (Tabela 1, linha ``iris``, coluna ``CBA (CARs)``; ``minconf`` 50%, ``minsup`` 1%,
discretização por entropia de Fayyad e Irani, validação cruzada em 10 dobras, média das dobras):

    erro 7,1% (sem e com poda) | nº de CARs: 72 (sem poda) e 23 (com poda, prCARs) | regras no classificador: 5

Os desvios da configuração atual em relação ao artigo são medidos um a um:

* **discretização**: quantis em 3 faixas (baseline) -> entropia/MDL (``mdl``);
* **minsup**: 5% (Iris) / 3% (Pokémon) -> 1%;
* **comprimento do antecedente**: limitado a 3 -> sem limite;
* **poda pessimista de condições** (opcional no artigo): desligada -> ligada.

Iris: 10 dobras (como o artigo), seeds 0--9. Pokémon (checagem de SENSIBILIDADE, sem número
publicado para comparar): 5 dobras por espécie, seeds 0--9. Execuções em memória; nada é gravado
em ``resultados/``. Saída: ``docs/relatorio_baseline/fidelidade_paper.json``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import pandas as pd

from experimentos.runner import REPO_ROOT, load_config, run_experiment
from fase2_implementacao.discretization_mdl import MDLDiscretizer

OUT = REPO_ROOT / "docs" / "relatorio_baseline" / "fidelidade_paper.json"
SEEDS = list(range(10))

PAPER_IRIS = {"error": 0.071, "cars": 72, "prcars": 23, "rules_in_c": 5, "folds": 10, "minsup": 0.01, "minconf": 0.5}

# nome, rótulo, overrides
IRIS_VARIANTS = [
    ("base", "Baseline atual (quantis, minsup 5\\%, antecedente $\\leq 3$)", dict(discretization_method="quantile", min_support=0.05, max_antecedent_length=3)),
    ("so_mdl", "Troca só a discretização (entropia/MDL)", dict(discretization_method="mdl", min_support=0.05, max_antecedent_length=3)),
    ("mdl_1pct", "MDL + minsup 1\\%", dict(discretization_method="mdl", min_support=0.01, max_antecedent_length=3)),
    ("paper", "Condições do artigo (MDL, 1\\%, sem limite) --- CARs", dict(discretization_method="mdl", min_support=0.01, max_antecedent_length=None)),
    ("paper_poda", "Condições do artigo + poda pessimista --- prCARs", dict(discretization_method="mdl", min_support=0.01, max_antecedent_length=None, prune=True)),
    ("quant_1pct", "Quantis, minsup 1\\%, sem limite (isola a discretização)", dict(discretization_method="quantile", min_support=0.01, max_antecedent_length=None)),
]
POKEMON_VARIANTS = [
    ("base", "Baseline atual (quantis, minsup 3\\%, antecedente $\\leq 3$)", dict(discretization_method="quantile", min_support=0.03, max_antecedent_length=3)),
    ("so_mdl", "Troca só a discretização (MDL)", dict(discretization_method="mdl", min_support=0.03, max_antecedent_length=3)),
    ("mdl_1pct", "MDL + minsup 1\\%", dict(discretization_method="mdl", min_support=0.01, max_antecedent_length=3)),
    ("paper", "Condições do artigo (MDL, 1\\%, sem limite)", dict(discretization_method="mdl", min_support=0.01, max_antecedent_length=None)),
    ("paper_poda", "Condições do artigo + poda pessimista", dict(discretization_method="mdl", min_support=0.01, max_antecedent_length=None, prune=True)),
    ("quant_1pct", "Quantis, minsup 1\\%, sem limite", dict(discretization_method="quantile", min_support=0.01, max_antecedent_length=None)),
]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def run_variants(yaml_name: str, variants, extra: dict) -> dict:
    out = {}
    for name, label, overrides in variants:
        per_seed = []
        for seed in SEEDS:
            config = load_config(REPO_ROOT / "experimentos" / "configs" / f"{yaml_name}.yaml")
            config.update(extra)
            config.update(overrides)
            config["seed"] = seed
            start = time.perf_counter()
            summary = run_experiment(config)["summary"]
            oof = summary["oof_metrics"]
            per_seed.append(
                {
                    "seed": seed,
                    "error": 1 - oof["accuracy"],
                    "accuracy": oof["accuracy"],
                    "balanced_accuracy": oof["balanced_accuracy"],
                    "mcc": oof["mcc"],
                    "recall_positive": summary["oof_recall_positive_class"],
                    "cars": summary["num_candidate_rules_mean"],
                    "rules_in_c": summary["num_final_rules_mean"],
                    "avg_antecedent_length": summary["avg_antecedent_length_mean"],
                    "seconds": time.perf_counter() - start,
                }
            )
        out[name] = {
            "label": label,
            "overrides": {k: v for k, v in overrides.items()},
            "per_seed": per_seed,
            "agg": {k: stats([r[k] for r in per_seed]) for k in per_seed[0] if k != "seed"},
        }
        a = out[name]["agg"]
        print(f"[{yaml_name}] {name}: erro {100 * a['error']['mean']:.2f}% | CARs {a['cars']['mean']:.1f} | regras {a['rules_in_c']['mean']:.1f}", flush=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    iris = run_variants("iris", IRIS_VARIANTS, {"k_folds": 10, "discretizar": True})
    pokemon = run_variants("pokemon", POKEMON_VARIANTS, {"discretizar": True})

    full = pd.read_csv(REPO_ROOT / "datasets" / "iris.csv")
    cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    mdl = MDLDiscretizer(cols).fit(full[cols], full["Species"])
    data = {
        "paper_iris": PAPER_IRIS,
        "seeds": SEEDS,
        "iris": iris,
        "pokemon": pokemon,
        "iris_full_mdl_cuts": {c: [float(e) for e in mdl.edges_[c]] for c in cols},
    }
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("salvo em", OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
