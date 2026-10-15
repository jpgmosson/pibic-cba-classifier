"""Comparação da baseline com o pyARC (referência externa, MIT) nas MESMAS dobras.

    python -m experimentos.comparar_pyarc [--seeds 42 0 1 2 3 4]

O pyARC exige o ``pyfim`` (extensão em C), que precisa de compilador; por isso ele roda num
container Docker (``experimentos/pyarc_docker/Dockerfile``, imagem ``pyarc-compare``). A
discretização, as dobras e as nossas regras são geradas aqui (host) e entregues ao container
(``pyarc_side.py``); a comparação em si é feita aqui, em quatro níveis:

1. **Mineração**: as regras (antecedente, classe, contagem conjunta) que passam em minsup/minconf,
   antes de qualquer escolha, são as mesmas no nosso Apriori e no ``fim.apriori`` do pyARC?
2. **Regra possível (PR)**: aplicando ao pyARC a escolha de um consequente por antecedente do
   artigo, obtém-se as nossas CARs?
3. **CBA-CB isolado (v3)**: alimentando o M1 do pyARC com as NOSSAS CARs, o classificador e as
   predições coincidem com os nossos?
4. **Ponta a ponta (v0)**: pyARC puro (CBA.fit) versus a nossa baseline; e as variantes
   intermediárias v1 (sem antecedente vazio) e v2 (v1 + PR) para atribuir as diferenças.

Saída: ``docs/relatorio_baseline/comparacao_pyarc.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pandas as pd

from experimentos.metricas import compute_metrics
from experimentos.runner import (
    REPO_ROOT,
    load_config,
    stratified_group_kfold_indices,
    stratified_kfold_indices,
)
from fase2_implementacao.apriori import apriori_frequent_itemset_counts
from fase2_implementacao.association_rules import dataframe_to_feature_transactions, dataframe_to_transactions, to_fraction
from fase2_implementacao.cba import build_classifier_m1, generate_cars
from fase2_implementacao.discretization import EqualFrequencyDiscretizer

SEP = ":=:"
IMAGE = "pyarc-compare"
OUT = REPO_ROOT / "docs" / "relatorio_baseline" / "comparacao_pyarc.json"
CONFIGS = [("iris_disc", "iris", "row"), ("pokemon_disc_group", "pokemon", "group"), ("pokemon_disc_row", "pokemon", "row")]


def item_str(item) -> str:
    return f"{item[0]}{SEP}{item[1]}"


def build_cases(seeds: list[int]) -> dict:
    cases = {}
    for name, yaml_name, split in CONFIGS:
        for seed in seeds:
            config = load_config(REPO_ROOT / "experimentos" / "configs" / f"{yaml_name}.yaml")
            df = pd.read_csv(REPO_ROOT / config["dataset_path"], sep=config["csv_sep"])
            cls, idc, grp = config["class_column"], config["id_column"], config["group_column"]
            feats = config["feature_columns"] or [c for c in df.columns if c not in (cls, idc, grp)]
            labels = df[cls].astype(str).tolist()
            if split == "group":
                folds = stratified_group_kfold_indices(labels, df[grp].astype(str).tolist(), config["k_folds"], seed)
            else:
                folds = stratified_kfold_indices(labels, config["k_folds"], seed)
            model_df = df[feats + [cls]]
            for k, test_idx in enumerate(folds):
                test_set = set(test_idx)
                train_df = model_df.iloc[[i for i in range(len(df)) if i not in test_set]].reset_index(drop=True)
                test_df = model_df.iloc[test_idx].reset_index(drop=True)
                disc = EqualFrequencyDiscretizer(config["n_bins"], config["continuous_columns"])
                train_enc, test_enc = disc.fit_transform(train_df), disc.transform(test_df)

                train_tx = dataframe_to_transactions(train_enc, cls)
                test_tx = dataframe_to_feature_transactions(test_enc, cls)
                y_train = train_df[cls].astype(str).tolist()
                maxlen_ant = config["max_antecedent_length"]
                cars = generate_cars(train_tx, cls, config["min_support"], config["min_confidence"], maxlen_ant)
                clf = build_classifier_m1(cars, train_tx, y_train, cls)

                counts, _, n = apriori_frequent_itemset_counts(train_tx, config["min_support"], maxlen_ant + 1)
                min_conf = to_fraction(config["min_confidence"])
                all_items = []
                for itemset, joint in counts.items():
                    ci = [it for it in itemset if it[0] == cls]
                    if len(ci) != 1 or len(itemset) < 2:
                        continue
                    ant = itemset - {ci[0]}
                    if Fraction(joint, counts[ant]) >= min_conf:
                        all_items.append([sorted(item_str(i) for i in ant), item_str(ci[0]), joint, counts[ant]])

                cases[f"{name}|{seed}|{k}"] = {
                    "header": list(train_enc.columns),
                    "train": train_enc.astype(str).values.tolist(),
                    "test": test_enc.astype(str).values.tolist(),
                    "minsup": config["min_support"],
                    "minconf": config["min_confidence"],
                    "maxlen": maxlen_ant + 1,  # pyARC conta o item de classe
                    "n_train": n,
                    "our_cars": [
                        {"ant": sorted(item_str(i) for i in c.antecedent), "con": item_str(c.consequent),
                         "support": c.support, "confidence": c.confidence, "joint": c.joint_count}
                        for c in cars
                    ],
                    "all_ruleitems": all_items,
                    "our_rules": [[sorted(item_str(i) for i in r.antecedent), item_str(r.consequent)] for r in clf.rules],
                    "our_default": str(clf.default_class),
                    "our_pred": [str(clf.predict(t)) for t in test_tx],
                    "y_true": test_df[cls].astype(str).tolist(),
                    "m1_prefix": clf.chosen_prefix_length,
                }
    return cases


def run_container(cases: dict) -> dict:
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        print("construindo a imagem Docker do pyARC...", flush=True)
        subprocess.run(["docker", "build", "-t", IMAGE, str(REPO_ROOT / "experimentos" / "pyarc_docker")], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "input.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
        shutil.copy(REPO_ROOT / "experimentos" / "pyarc_side.py", work / "pyarc_side.py")
        subprocess.run(["docker", "run", "--rm", "-v", f"{work}:/work", IMAGE, "python", "/work/pyarc_side.py"], check=True)
        return json.loads((work / "pyarc.json").read_text(encoding="utf-8"))


def key_of(rule_ant, con):
    return (tuple(sorted(rule_ant)), con)


def compare_case(case: dict, py: dict) -> dict:
    n = case["n_train"]
    ours_all = {key_of(a, c): j for a, c, j, _ in case["all_ruleitems"]}
    py_nonempty = {key_of(r["ant"], r["con"]): round(r["support"] * n) for r in py["pyarc_cars"] if r["ant"]}
    only_ours = sorted(set(ours_all) - set(py_nonempty))
    only_py = sorted(set(py_nonempty) - set(ours_all))
    count_diff = [k for k in set(ours_all) & set(py_nonempty) if ours_all[k] != py_nonempty[k]]

    # PR aplicada ao pyARC (host): maior confiança por antecedente; empate: menor rótulo
    best = {}
    for r in py["pyarc_cars"]:
        if not r["ant"]:
            continue
        k = tuple(r["ant"])
        if k not in best or (-r["confidence"], r["con"]) < (-best[k]["confidence"], best[k]["con"]):
            best[k] = r
    py_pr = {key_of(r["ant"], r["con"]) for r in best.values()}
    our_pr = {key_of(c["ant"], c["con"]) for c in case["our_cars"]}

    out = {
        "n_ruleitems_ours": len(ours_all), "n_ruleitems_pyarc": len(py_nonempty),
        "n_empty_antecedent_pyarc": py["n_empty_antecedent"],
        "miner_only_ours": len(only_ours), "miner_only_pyarc": len(only_py), "miner_count_mismatch": len(count_diff),
        "miner_identical": not only_ours and not only_py and not count_diff,
        "pr_identical": py_pr == our_pr, "n_cars_ours": len(our_pr), "n_cars_pyarc_pr": len(py_pr),
        "only_ours_examples": [list(k) for k in only_ours[:3]], "only_pyarc_examples": [list(k) for k in only_py[:3]],
    }
    # Diagnóstico de empates: dentro do conjunto de CARs (PR), regras com a MESMA confiança exata podem ter
    # confianças em ponto flutuante diferentes no fim/pyARC (achado A10 em dados reais).
    counts = {key_of(a, c): (j, ac) for a, c, j, ac in case["all_ruleitems"]}
    groups = defaultdict(list)
    for r in best.values():
        j, ac = counts[key_of(r["ant"], r["con"])]
        groups[Fraction(j, ac)].append(r["confidence"])
    out["tied_conf_groups"] = sum(1 for g in groups.values() if len(g) > 1)
    out["tied_conf_groups_with_float_noise"] = sum(1 for g in groups.values() if len(set(g)) > 1)
    out["rules_in_noisy_ties"] = sum(len(g) for g in groups.values() if len(set(g)) > 1)

    def prec_key(a, c):
        j, ac = counts[key_of(a, c)]
        return (Fraction(j, ac), j, len(a))

    y = case["y_true"]
    for v in ("v0", "v1", "v2", "v3"):
        r = py[v]
        same_rules = [key_of(a, c) for a, c in r["rules"]] == [key_of(a, c) for a, c in case["our_rules"]]
        agree = sum(1 for a, b in zip(r["pred"], case["our_pred"]) if a == b)
        set_same = {key_of(a, c) for a, c in r["rules"]} == {key_of(a, c) for a, c in case["our_rules"]}
        known = all(key_of(a, c) in counts for a, c in r["rules"])
        order_only_ties = (known and [prec_key(a, c) for a, c in r["rules"]] == [prec_key(a, c) for a, c in case["our_rules"]])
        out[v] = {
            "rule_set_identical": set_same, "differs_only_in_tie_order": (not same_rules) and order_only_ties,
            "classifier_identical": same_rules and r["default"] == case["our_default"],
            "rules_identical": same_rules, "default_identical": r["default"] == case["our_default"],
            "n_rules": len(r["rules"]), "n_rules_ours": len(case["our_rules"]),
            "pred_agreement": agree, "n_test": len(y), "pyarc_default": r["default"], "our_default": case["our_default"],
            "pyarc_correct": sum(1 for a, t in zip(r["pred"], y) if a == t),
        }
    out["ours_correct"] = sum(1 for a, t in zip(case["our_pred"], y) if a == t)
    # v0, v1 e v2 (com/sem antecedente vazio, com/sem PR) produzem o MESMO classificador e as MESMAS predições?
    sig = lambda r: ([key_of(a, c) for a, c in r["rules"]], r["default"], r["pred"])  # noqa: E731
    out["v0_v1_v2_identical"] = sig(py["v0"]) == sig(py["v1"]) == sig(py["v2"])
    return out


def aggregate(cases: dict, py: dict, per_case: dict) -> dict:
    by_cfg = defaultdict(list)
    for key in cases:
        by_cfg[key.split("|")[0]].append(key)
    summary = {}
    for cfg, keys in by_cfg.items():
        s = {"folds": len(keys), "seeds": sorted({int(k.split("|")[1]) for k in keys})}
        for flag in ("miner_identical", "pr_identical"):
            s[flag] = sum(per_case[k][flag] for k in keys)
        s["v0_v1_v2_identical_folds"] = sum(per_case[k]["v0_v1_v2_identical"] for k in keys)
        s["tied_conf_groups"] = sum(per_case[k]["tied_conf_groups"] for k in keys)
        s["tied_conf_groups_with_float_noise"] = sum(per_case[k]["tied_conf_groups_with_float_noise"] for k in keys)
        s["rules_in_noisy_ties"] = sum(per_case[k]["rules_in_noisy_ties"] for k in keys)
        s["empty_antecedent_rules_total"] = sum(per_case[k]["n_empty_antecedent_pyarc"] for k in keys)
        s["ruleitems_ours_total"] = sum(per_case[k]["n_ruleitems_ours"] for k in keys)
        s["ruleitems_pyarc_total"] = sum(per_case[k]["n_ruleitems_pyarc"] for k in keys)
        n_test = sum(per_case[k]["v0"]["n_test"] for k in keys)
        s["n_test_total"] = n_test
        s["ours_accuracy"] = sum(per_case[k]["ours_correct"] for k in keys) / n_test
        for v in ("v0", "v1", "v2", "v3"):
            s[v] = {
                "classifier_identical_folds": sum(per_case[k][v]["classifier_identical"] for k in keys),
                "rules_identical_folds": sum(per_case[k][v]["rules_identical"] for k in keys),
                "rule_set_identical_folds": sum(per_case[k][v]["rule_set_identical"] for k in keys),
                "differs_only_in_tie_order_folds": sum(per_case[k][v]["differs_only_in_tie_order"] for k in keys),
                "default_identical_folds": sum(per_case[k][v]["default_identical"] for k in keys),
                "pred_agreement": sum(per_case[k][v]["pred_agreement"] for k in keys) / n_test,
                "pyarc_accuracy": sum(per_case[k][v]["pyarc_correct"] for k in keys) / n_test,
                "mean_rules_pyarc": sum(per_case[k][v]["n_rules"] for k in keys) / len(keys),
                "mean_rules_ours": sum(per_case[k][v]["n_rules_ours"] for k in keys) / len(keys),
            }
        # métricas OOF por variante (sobre todas as dobras/seeds agregadas)
        y = [t for k in keys for t in cases[k]["y_true"]]
        labels = sorted(set(y))
        s["ours_metrics"] = {m: compute_metrics(y, [p for k in keys for p in cases[k]["our_pred"]], labels)[m] for m in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")}
        for v in ("v0", "v3"):
            s[v]["metrics"] = {m: compute_metrics(y, [p for k in keys for p in py[k][v]["pred"]], sorted(set(y) | {p for k in keys for p in py[k][v]["pred"]}))[m] for m in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")}
        summary[cfg] = s
    return summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3, 4])
    args = parser.parse_args()
    print("exportando dobras e regras...", flush=True)
    cases = build_cases(args.seeds)
    print(f"{len(cases)} dobras; rodando o pyARC no container...", flush=True)
    py = run_container(cases)
    per_case = {k: compare_case(cases[k], py[k]) for k in cases}
    summary = aggregate(cases, py, per_case)
    OUT.write_text(json.dumps({"seeds": args.seeds, "summary": summary, "per_fold": per_case}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
