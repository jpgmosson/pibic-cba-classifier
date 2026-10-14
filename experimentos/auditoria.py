"""Auditoria de uma execução do runner a partir **apenas dos artefatos salvos**.

Nada aqui chama o CBA, o Apriori ou o discretizador: tudo é reconstruído de
``config.json``, ``discretization.json``, ``rules.json``, ``m1_history.json``,
``partitions.json`` e ``oof_predictions.csv`` (mais o CSV original do dataset).
É o critério de aceite do Relatório 02: explicar uma predição do registro
original até a classe, mostrar o corte do M1 e reproduzir as métricas.

    python -m experimentos.auditoria verify  --run experimentos/resultados/<run_id>
    python -m experimentos.auditoria history --run <dir> [--fold 0]
    python -m experimentos.auditoria trace   --run <dir> --row <row_id> [--row-id-is-name]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.metricas import compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


class Run:
    """Artefatos de uma execução, carregados do disco."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        load = lambda name: json.loads((self.dir / name).read_text(encoding="utf-8"))  # noqa: E731
        self.prov = load("config.json")
        self.config = self.prov["config"]
        self.summary = load("summary.json")
        self.rules = load("rules.json")
        self.history = load("m1_history.json")
        self.discretization = load("discretization.json")
        self.partitions = load("partitions.json")
        with open(self.dir / "oof_predictions.csv", encoding="utf-8", newline="") as f:
            self.oof = list(csv.DictReader(f))
        self.df = pd.read_csv(REPO_ROOT / self.config["dataset_path"], sep=self.config["csv_sep"])

    def items_for_row(self, row_id: int, fold: int) -> tuple[list[str], dict]:
        """Itens ``atributo=valor`` que o classificador enxerga para a linha (sem a classe)."""
        row = self.df.iloc[row_id]
        disc = self.discretization.get(str(fold))
        shown, items = {}, []
        for col in self.config["feature_columns"]:
            raw = row[col]
            if disc and col in disc:
                edges = np.array(disc[col]["edges"], dtype=float)
                value = f"bin{int(np.searchsorted(edges, float(raw), side='right'))}"
            else:
                value = self.df[col].astype(str).iloc[row_id]
            shown[col] = (raw.item() if hasattr(raw, "item") else raw, value)
            items.append(f"{col}={value}")
        return items, shown

    def final_rules(self, fold: int) -> list[dict]:
        rules = [r for r in self.rules[str(fold)] if r["final_position"] is not None]
        return sorted(rules, key=lambda r: r["final_position"])

    def reconstruct(self, row_id: int, fold: int) -> tuple[str, int | None]:
        items, _ = self.items_for_row(row_id, fold)
        item_set = set(items)
        for rule in self.final_rules(fold):
            if set(rule["antecedent"]) <= item_set:
                return rule["consequent"].split("=", 1)[1], rule["final_position"]
        return str(self.history[str(fold)]["final_default_class"]), None


def verify(run: Run) -> bool:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= condition
        print(f"[{'OK' if condition else 'FALHOU'}] {label}{(' — ' + detail) if detail else ''}")

    n = len(run.df)
    rows = [int(r["row_id"]) for r in run.oof]
    check("cada linha do dataset tem exatamente uma predição OOF", sorted(rows) == list(range(n)), f"{len(rows)} predições, N={n}")

    mismatches = []
    for r in run.oof:
        label, position = run.reconstruct(int(r["row_id"]), int(r["fold"]))
        recorded = int(r["rule_index"]) if r["rule_index"] != "" else None
        if label != r["y_pred"] or position != recorded:
            mismatches.append((r["row_id"], r["y_pred"], label))
    check(
        "predições reconstruídas só de cortes + regras salvas == predições OOF",
        not mismatches,
        f"{len(run.oof) - len(mismatches)}/{len(run.oof)} idênticas",
    )

    labels = sorted({r["y_true"] for r in run.oof} | {r["y_pred"] for r in run.oof})
    recomputed = compute_metrics([r["y_true"] for r in run.oof], [r["y_pred"] for r in run.oof], labels=labels)
    saved = run.summary["oof_metrics"]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mcc"):
        check(f"métrica OOF recalculada '{key}' == resumo", abs(recomputed[key] - saved[key]) < 1e-12, f"{recomputed[key]:.6f}")
    check("matriz de confusão OOF recalculada == resumo", recomputed["confusion_matrix"]["matrix"] == saved["confusion_matrix"]["matrix"])

    maj = run.summary["majority_baseline_metrics"]["accuracy"]
    print(f"     baseline classe majoritária (OOF): acurácia {maj:.4f}  |  CBA (OOF): acurácia {recomputed['accuracy']:.4f}")

    for fold, hist in run.history.items():
        totals = [p["total_errors"] for p in hist["prefixes"]]
        if totals:
            check(
                f"fold {fold}: prefixo do M1 escolhido é o primeiro de erro total mínimo",
                hist["chosen_prefix_length"] == totals.index(min(totals)) + 1
                and len(run.final_rules(int(fold))) == hist["chosen_prefix_length"],
                f"{hist['chosen_prefix_length']} regras, erro total {min(totals)}",
            )
    shared = [p.get("groups_shared_train_test") for p in run.partitions if "groups_shared_train_test" in p]
    if shared:
        print(f"     grupos compartilhados entre treino e teste por fold: {shared}")
    print("RESULTADO:", "APROVADO" if ok else "REPROVADO")
    return ok


def show_history(run: Run, fold: int) -> None:
    hist = run.history[str(fold)]
    print(f"Histórico de erro do M1 — fold {fold} (run {run.prov['run_id']})")
    print(f"{'k':>3} {'erros regras':>12} {'residuais':>9} {'padrão':>8} {'erros padrão':>12} {'erro total':>10}  regra adicionada")
    for p in hist["prefixes"]:
        mark = "  <== PREFIXO ESCOLHIDO (1º mínimo)" if p["chosen"] else ""
        print(f"{p['n_rules']:>3} {p['rule_errors']:>12} {p['n_remaining']:>9} {str(p['default_class']):>8} {p['default_errors']:>12} {p['total_errors']:>10}  {p['rule']}{mark}")
    print(f"Classificador final: {hist['chosen_prefix_length']} regra(s) + padrão {hist['final_default_class']!r}")


def trace(run: Run, row_id: int) -> None:
    record = next(r for r in run.oof if int(r["row_id"]) == row_id)
    fold = int(record["fold"])
    items, shown = run.items_for_row(row_id, fold)
    print(f"=== Rastreamento: linha {row_id} ({record['id']}) — fold {fold} — run {run.prov['run_id']}")
    print(f"Classe verdadeira: {record['y_true']}   (usada só para avaliar; nunca na inferência)")
    print("Preditor bruto -> item visto pelo classificador:")
    disc = run.discretization.get(str(fold)) or {}
    for col, (raw, value) in shown.items():
        extra = f"   cortes de treino do fold: {[round(e, 3) for e in disc[col]['edges']]}" if col in disc else ""
        print(f"  {col} = {raw}  ->  {col}={value}{extra}")

    item_set = set(items)
    matching = [r for r in run.rules[str(fold)] if set(r["antecedent"]) <= item_set]
    matching.sort(key=lambda r: (r["precedence_position"] is None, r["precedence_position"]))
    print(f"\nRegras CANDIDATAS compatíveis com o caso ({len(matching)} de {len(run.rules[str(fold)])} candidatas do fold), por precedência:")
    for r in matching[:12]:
        print(
            f"  #{r['precedence_position']:<4} conf {r['confidence']:>7} ({r['confidence_float']:.3f})  sup {r['support_float']:.3f}  "
            f"[{r['status']}]  {r['text']}"
        )
    if len(matching) > 12:
        print(f"  ... (+{len(matching) - 12} regras compatíveis menos precedentes)")

    label, position = run.reconstruct(row_id, fold)
    hist = run.history[str(fold)]
    if position is None:
        print(f"\nNenhuma regra FINAL cobre o caso -> classe padrão {label!r} (padrão do prefixo escolhido do M1).")
    else:
        fired = run.final_rules(fold)[position]
        print(f"\nRegra ACIONADA (1ª compatível na lista final, posição {position}): {fired['text']}")
        print(f"  confiança {fired['confidence']} = {fired['confidence_float']:.3f}; suporte {fired['support_float']:.3f}; precedência #{fired['precedence_position']}")
        print(f"  -> classe prevista {label!r}")
    cut = [r for r in matching if r["status"] == "selected_then_cut_by_m1"]
    if cut:
        first = cut[0]
        consequence = (
            f"Sem o corte, ela decidiria este caso (classe {first['consequent'].split('=', 1)[1]!r}) em vez do padrão."
            if position is None
            else "Ela não altera esta decisão: uma regra de maior precedência, mantida no prefixo, já cobre o caso."
        )
        print(
            f"\nCorte do M1: foram selecionadas {len(hist['prefixes'])} regras por cobertura, mas só as "
            f"{hist['chosen_prefix_length']} primeiras foram mantidas (erro total mínimo). A regra compatível "
            f"'{first['text']}' (selecionada na posição {first['selection_position'] + 1}) foi cortada. {consequence}"
        )
    print(f"Prefixo escolhido: {hist['chosen_prefix_length']} regras (erro total mínimo = "
          f"{min((p['total_errors'] for p in hist['prefixes']), default='n/d')}), padrão {hist['final_default_class']!r}.")
    print(f"\nPredição registrada no OOF: {record['y_pred']!r}  | reconstruída dos artefatos: {label!r}  | "
          f"{'CORRETA' if record['y_pred'] == record['y_true'] else 'INCORRETA'}  | "
          f"{'BATEM' if label == record['y_pred'] else 'DIVERGEM'}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["verify", "history", "trace"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--row", type=int)
    args = parser.parse_args(argv)
    run = Run(args.run)
    if args.command == "verify":
        sys.exit(0 if verify(run) else 1)
    elif args.command == "history":
        show_history(run, args.fold)
    else:
        if args.row is None:
            parser.error("trace exige --row")
        trace(run, args.row)


if __name__ == "__main__":
    main()
