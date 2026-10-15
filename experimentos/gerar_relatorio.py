"""Roda a bateria completa de testes da baseline (Iris e Pokémon) e gera os
dados e as tabelas LaTeX do relatório final.

    python -m experimentos.gerar_relatorio            # só baseline atual
    python -m experimentos.gerar_relatorio --com-antigo   # + código anterior (commit fdc8cb6)

Saídas (em ``docs/relatorio_baseline/``):

- ``dados_baseline.json``  — todos os números usados no relatório;
- ``relatorio_baseline.tex`` — o relatório em UM único arquivo autocontido (tabelas embutidas), pronto para o Overleaf;

Protocolo: 6 configurações (Iris com/sem discretização; Pokémon com/sem
discretização × divisão por espécie/por linha), seed principal 42 (com
matrizes de confusão, folds, regras e histórico do M1) e varredura de 10
seeds (0--9) para medir a variabilidade. Nada é gravado em ``resultados/``
nem em ``resumo.csv``: são execuções em memória.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from experimentos.runner import REPO_ROOT, load_config, run_experiment

OUT_DIR = REPO_ROOT / "docs" / "relatorio_baseline"
SEEDS = list(range(10))
MAIN_SEED = 42
OLD_COMMIT = "fdc8cb6"

CONFIGS = [
    # nome, yaml, discretizar, split
    ("iris_disc", "iris", True, "row"),
    ("iris_raw", "iris", False, "row"),
    ("pokemon_disc_group", "pokemon", True, "group"),
    ("pokemon_disc_row", "pokemon", True, "row"),
    ("pokemon_raw_group", "pokemon", False, "group"),
    ("pokemon_raw_row", "pokemon", False, "row"),
]

LABELS = {
    "iris_disc": ("Iris", "discretizado", "linha"),
    "iris_raw": ("Iris", "bruto", "linha"),
    "pokemon_disc_group": ("Pokémon", "discretizado", "espécie"),
    "pokemon_disc_row": ("Pokémon", "discretizado", "linha"),
    "pokemon_raw_group": ("Pokémon", "bruto", "espécie"),
    "pokemon_raw_row": ("Pokémon", "bruto", "linha"),
}


def build_config(yaml_name: str, discretize: bool, split: str, seed: int) -> dict:
    config = load_config(REPO_ROOT / "experimentos" / "configs" / f"{yaml_name}.yaml")
    config["discretizar"] = discretize
    config["seed"] = seed
    if config["group_column"] is not None:
        config["split"] = split
    return config


def compact(run: dict, detailed: bool) -> dict:
    s, a = run["summary"], run["artifacts"]
    oof, base = s["oof_metrics"], s["majority_baseline_metrics"]
    pos = s["config"]["positive_class"]
    out = {
        "oof": {k: oof[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")},
        "confusion": oof["confusion_matrix"],
        "per_class": oof["per_class"],
        "majority": {k: base[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")},
        "recall_positive_oof": s["oof_recall_positive_class"],
        "recall_positive_fold_mean": s["recall_positive_class_mean"],
        "accuracy_fold_mean": s["accuracy_mean"],
        "accuracy_fold_std": s["accuracy_std"],
        "uncertainty": s["oof_uncertainty"],
        "num_candidate_rules_mean": s["num_candidate_rules_mean"],
        "num_final_rules_mean": s["num_final_rules_mean"],
        "avg_antecedent_length_mean": s["avg_antecedent_length_mean"],
        "default_usage_rate_mean": s["default_class_usage_rate_mean"],
        "total_seconds": s["total_elapsed_seconds"],
        "positive_class": pos,
    }
    if detailed:
        out["folds"] = s["folds"]
        out["partitions"] = [{k: v for k, v in p.items() if k not in ("test_row_ids", "test_ids")} for p in a["partitions"]]
        out["discretization_fold0"] = a["discretization"].get(0)
        out["history"] = a["m1_history"]
        out["final_rules_fold0"] = sorted(
            (r for r in a["rules"][0] if r["final_position"] is not None), key=lambda r: r["final_position"]
        )
        out["n_candidates_fold"] = {k: len(v) for k, v in a["rules"].items()}
        out["rule_status_fold0"] = {
            st: sum(1 for r in a["rules"][0] if r["status"] == st)
            for st in ("final", "selected_then_cut_by_m1", "not_selected_by_coverage")
        }
    return out


def run_all() -> dict:
    data = {"main_seed": MAIN_SEED, "seeds": SEEDS, "main": {}, "sweep": {}}
    for name, yaml_name, disc, split in CONFIGS:
        print(f"[{name}] seed {MAIN_SEED}", flush=True)
        data["main"][name] = compact(run_experiment(build_config(yaml_name, disc, split, MAIN_SEED)), detailed=True)
        data["sweep"][name] = {}
        for seed in SEEDS:
            data["sweep"][name][seed] = compact(run_experiment(build_config(yaml_name, disc, split, seed)), detailed=False)
        print(f"[{name}] varredura de {len(SEEDS)} seeds concluída", flush=True)
    return data


OLD_SCRIPT = r"""
import json, sys
from pathlib import Path
from experimentos import runner
out = {}
for name, yaml_name, disc in [("iris_disc","iris",True),("iris_raw","iris",False),("pokemon_disc","pokemon",True),("pokemon_raw","pokemon",False)]:
    out[name] = {}
    for seed in [42] + list(range(10)):
        cfg = runner.load_config(Path("experimentos/configs")/(yaml_name+".yaml"))
        cfg["discretizar"] = disc
        cfg["seed"] = seed
        cfg["cb_variant"] = "m1"
        s = runner.run_experiment(cfg)
        out[name][str(seed)] = {"accuracy_mean": s["accuracy_mean"], "recall": s["recall_positive_class_mean"],
                                "final_rules": s["num_final_rules_mean"], "default_usage": s["default_class_usage_rate_mean"]}
print(json.dumps(out))
"""


def run_old() -> dict:
    """Roda o código anterior (commit OLD_COMMIT) num worktree temporário."""
    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "old"
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), OLD_COMMIT], cwd=REPO_ROOT, check=True, capture_output=True)
        try:
            result = subprocess.run(
                [sys.executable, "-c", OLD_SCRIPT], cwd=worktree, check=True, capture_output=True, text=True
            )
            return json.loads(result.stdout.strip().splitlines()[-1])
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO_ROOT, capture_output=True)


# ------------------------------------------------------------------ LaTeX ---


def esc(text) -> str:
    text = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(a, b)
    return text.replace(" => ", r" $\Rightarrow$ ")


def _round_half_up(x: float, d: int) -> str:
    """Arredondamento convencional (meio para cima) sobre o valor decimal de ``x``."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(repr(round(x, 12))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pct(x, d=2):
    return "--" if x is None else _round_half_up(100 * x, d).replace(".", ",") + r"\%"


def num(x, d=3):
    return "--" if x is None else _round_half_up(x, d).replace(".", ",")


def ms(values, d=2, percent=True):
    m, s = statistics.mean(values), statistics.pstdev(values)
    f = (lambda v: pct(v, d)) if percent else (lambda v: num(v, d))
    return f"{f(m)} $\\pm$ {f(s)}"


def table(caption, label, header, rows, spec, small=True):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small" if small else "", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(l for l in lines if l != "")


def build_tables(data: dict, old: dict | None) -> str:
    main, sweep = data["main"], data["sweep"]
    out = []

    def lab(n):
        d, t, s = LABELS[n]
        return f"{d} ({t}, por {s})"

    # 1) tabela principal
    rows = []
    for n, _, _, _ in CONFIGS:
        m = main[n]
        ci = m["uncertainty"]["bootstrap"]["ci"]["accuracy"]
        rows.append([lab(n), pct(m["oof"]["accuracy"]), f"[{pct(ci[0], 1)}; {pct(ci[1], 1)}]", pct(m["majority"]["accuracy"]),
                     num(m["oof"]["balanced_accuracy"]), num(m["oof"]["macro_f1"]), num(m["oof"]["mcc"]),
                     pct(m["recall_positive_oof"]) if m["recall_positive_oof"] is not None else "--"])
    out.append(table(f"Resultados da baseline (seed {MAIN_SEED}, predições fora da dobra agregadas). IC: bootstrap 95\\% (por espécie quando há grupos).",
                     "tab:principal", ["Configuração", "Acurácia", "IC 95\\%", "Majoritária", "Acc. bal.", "Macro-F1", "MCC", "Recall $+$"], rows, "lrrrrrrr"))

    # 2) matrizes de confusão
    for n, _, _, _ in CONFIGS:
        c = main[n]["confusion"]
        labels, mat = c["labels"], c["matrix"]
        rows = [[esc(labels[i])] + [str(v) for v in mat[i]] for i in range(len(labels))]
        out.append(table(f"Matriz de confusão OOF -- {lab(n)} (linhas: classe real; colunas: prevista).",
                         f"tab:conf_{n}", [r"Real $\backslash$ Prevista"] + [esc(l) for l in labels], rows, "l" + "r" * len(labels)))

    # 3) varredura de seeds
    rows = []
    for n, _, _, _ in CONFIGS:
        r = sweep[n]
        acc = [r[s]["oof"]["accuracy"] for s in data["seeds"]]
        mcc = [r[s]["oof"]["mcc"] for s in data["seeds"]]
        bal = [r[s]["oof"]["balanced_accuracy"] for s in data["seeds"]]
        maj = [r[s]["majority"]["accuracy"] for s in data["seeds"]]
        rec = [r[s]["recall_positive_oof"] for s in data["seeds"] if r[s]["recall_positive_oof"] is not None]
        rows.append([lab(n), ms(acc), pct(min(acc)) + "--" + pct(max(acc)), ms(bal, 3, False), ms(mcc, 3, False), ms(rec) if rec else "--", pct(statistics.mean(maj))])
    out.append(table(f"Variabilidade entre {len(data['seeds'])} seeds (0--9): média $\\pm$ desvio-padrão populacional das métricas OOF.",
                     "tab:seeds", ["Configuração", "Acurácia", "Mín--máx", "Acc. bal.", "MCC", "Recall $+$", "Majoritária"], rows, "lrrrrrr"))

    # 4) espécie vs linha (pareado por seed)
    rows = []
    for d in ("disc", "raw"):
        g, r = sweep[f"pokemon_{d}_group"], sweep[f"pokemon_{d}_row"]
        for key, nome, pc in (("accuracy", "Acurácia", True), ("mcc", "MCC", False)):
            diff = [g[s]["oof"][key] - r[s]["oof"][key] for s in data["seeds"]]
            wins = sum(1 for x in diff if x < 0)
            rows.append(["discretizado" if d == "disc" else "bruto", nome, ms([r[s]["oof"][key] for s in data["seeds"]], 3 if not pc else 2, pc),
                         ms([g[s]["oof"][key] for s in data["seeds"]], 3 if not pc else 2, pc),
                         (pct(statistics.mean(diff)) if pc else num(statistics.mean(diff), 4)), f"{wins}/{len(diff)}"])
    out.append(table("Efeito de separar espécies (Pokémon): divisão por linha versus por espécie, mesmas seeds. Diferença = espécie $-$ linha.",
                     "tab:grupo_linha", ["Atributos", "Métrica", "Por linha", "Por espécie", "Diferença média", "Seeds com espécie $<$ linha"], rows, "llrrrr"))

    # 5) antigo vs novo
    if old:
        rows = []
        pairs = [("iris_disc", "iris_disc", "Iris (disc.)"), ("iris_raw", "iris_raw", "Iris (bruto)"),
                 ("pokemon_disc", "pokemon_disc_row", "Pokémon (disc., linha)"), ("pokemon_raw", "pokemon_raw_row", "Pokémon (bruto, linha)")]
        for oname, nname, title in pairs:
            o42, n42 = old[oname]["42"], main[nname]
            o_acc = [old[oname][str(s)]["accuracy_mean"] for s in data["seeds"]]
            n_acc = [sweep[nname][s]["accuracy_fold_mean"] for s in data["seeds"]]
            o_rec = [old[oname][str(s)]["recall"] for s in data["seeds"] if old[oname][str(s)]["recall"] is not None]
            n_rec = [sweep[nname][s]["recall_positive_fold_mean"] for s in data["seeds"] if sweep[nname][s]["recall_positive_fold_mean"] is not None]
            rows.append([title, pct(o42["accuracy_mean"]), pct(n42["accuracy_fold_mean"]), ms(o_acc), ms(n_acc),
                         (pct(o42["recall"]) if o42["recall"] is not None else "--") + " / " + (pct(n42["recall_positive_fold_mean"]) if n42["recall_positive_fold_mean"] is not None else "--"),
                         (ms(o_rec) + " / " + ms(n_rec)) if o_rec else "--"])
        out.append(table(f"Código anterior (commit {OLD_COMMIT}) versus baseline corrigida, mesma divisão por linha. Acurácia: média das acurácias por dobra (definição do código antigo).",
                         "tab:antigo_novo", ["Experimento", "Acc. antiga (42)", "Acc. nova (42)", "Acc. antiga (10 seeds)", "Acc. nova (10 seeds)", "Recall $+$ ant./nova (42)", "Recall $+$ ant./nova (10 seeds)"],
                         rows, "lrrrrrr"))

    # 6) folds do Pokémon por espécie
    m = main["pokemon_disc_group"]
    rows = []
    for p, f in zip(m["partitions"], m["folds"]):
        rows.append([str(p["fold"]), str(p["train_rows"]), str(p["test_rows"]), f"{p['test_class_counts'].get('True', 0)}/{p['test_class_counts'].get('False', 0)}",
                     str(p["test_groups"]), str(p["groups_shared_train_test"]), pct(f["accuracy"]), pct(f["recall_positive_class"]), str(f["num_candidate_rules"]), str(f["num_final_rules"]), str(f["m1_chosen_prefix_length"])])
    out.append(table(f"Pokémon discretizado, divisão por espécie (seed {MAIN_SEED}): composição e resultado de cada dobra.",
                     "tab:folds_pokemon", ["Dobra", "Treino", "Teste", "Teste Leg./Não", "Espécies teste", "Espécies compart.", "Acurácia", "Recall $+$", "Cand.", "Finais", "Prefixo M1"], rows, "rrrrrrrrrrr"))
    mr = main["pokemon_disc_row"]
    rows = [[str(p["fold"]), str(p["test_rows"]), str(p["groups_shared_train_test"]), pct(f["accuracy"]), pct(f["recall_positive_class"])]
            for p, f in zip(mr["partitions"], mr["folds"])]
    out.append(table(f"Pokémon discretizado, divisão por linha (seed {MAIN_SEED}): espécies presentes ao mesmo tempo em treino e teste.",
                     "tab:folds_linha", ["Dobra", "Teste", "Espécies compartilhadas", "Acurácia", "Recall $+$"], rows, "rrrrr"))
    mi = main["iris_disc"]
    rows = [[str(p["fold"]), str(p["train_rows"]), str(p["test_rows"]), pct(f["accuracy"]), str(f["num_candidate_rules"]), str(f["num_final_rules"]), str(f["m1_chosen_prefix_length"]), pct(f["default_class_usage_rate"])]
            for p, f in zip(mi["partitions"], mi["folds"])]
    out.append(table(f"Iris discretizado (seed {MAIN_SEED}): resultado por dobra.", "tab:folds_iris",
                     ["Dobra", "Treino", "Teste", "Acurácia", "Cand.", "Finais", "Prefixo M1", "Uso do padrão"], rows, "rrrrrrrr"))

    # 7) cortes de discretização
    for n, title in (("iris_disc", "Iris"), ("pokemon_disc_group", "Pokémon")):
        d = main[n]["discretization_fold0"]
        rows = [[esc(col), ", ".join(f"{e:g}".replace(".", ",") for e in v["edges"]), str(v["n_effective_bins"]),
                 " / ".join(str(c) for c in v["occupancy"].values())] for col, v in d.items()]
        out.append(table(f"Cortes de discretização por igual-frequência aprendidos no treino da dobra 0 -- {title} (seed {MAIN_SEED}); ocupação dos bins no treino.",
                         f"tab:cortes_{title.lower().replace('é', 'e').replace('ó','o')}", ["Atributo", "Cortes", "Bins", "Ocupação (bin0 / bin1 / bin2)"], rows, "lrrr"))

    # 8) histórico do M1
    def history_table(name, fold, title, label):
        h = main[name]["history"][fold]
        prefixes = h["prefixes"]
        keep = set(range(min(3, len(prefixes)))) | set(range(len(prefixes) - 2, len(prefixes)))
        chosen = h["chosen_prefix_length"] - 1
        keep |= set(range(max(0, chosen - 2), min(len(prefixes), chosen + 3)))
        rows, prev = [], -1
        for i in sorted(keep):
            if i - prev > 1:
                rows.append([r"\multicolumn{7}{c}{\ldots}"])
            p = prefixes[i]
            mark = r"\textbf{*} " if p["chosen"] else ""
            rows.append([mark + str(p["n_rules"]), str(p["rule_errors"]), str(p["n_remaining"]), esc(p["default_class"]), str(p["default_errors"]), str(p["total_errors"]),
                         esc(p["rule"]).replace(r"\_", r"\_")])
            prev = i
        return table(title + r" (* = prefixo escolhido: primeiro de erro total mínimo)", label,
                     ["$k$", "Erros regras", "Residuais", "Padrão", "Erros padrão", "Erro total", "Regra $k$ adicionada"], rows, "rrrrrrp{5.2cm}")
    out.append(history_table("iris_disc", 0, f"Histórico de erro do M1 -- Iris discretizado, dobra 0 (seed {MAIN_SEED})", "tab:m1_iris"))
    out.append(history_table("pokemon_disc_group", 3, f"Histórico de erro do M1 -- Pokémon discretizado (por espécie), dobra 3 (seed {MAIN_SEED})", "tab:m1_pokemon"))

    # 9) regras finais
    for n, title, k in (("iris_disc", "Iris", 8), ("pokemon_disc_group", "Pokémon", 8)):
        rows = [[str(r["final_position"] + 1), esc(r["text"]), r["confidence"], num(r["confidence_float"]), num(r["support_float"])] for r in main[n]["final_rules_fold0"][:k]]
        st = main[n]["rule_status_fold0"]
        out.append(table(f"Primeiras {k} regras do classificador final -- {title}, dobra 0 (seed {MAIN_SEED}). Na dobra: {st['final']} regras finais, {st['selected_then_cut_by_m1']} selecionadas e cortadas pelo M1, {st['not_selected_by_coverage']} não selecionadas pela cobertura.",
                         f"tab:regras_{title.lower().replace('é','e').replace('ó','o')}", ["Pos.", "Regra", "Conf. (fração)", "Conf.", "Sup."], rows, "rp{7.2cm}rrr"))

    # 10) tamanho e tempo
    rows = []
    for n, _, _, _ in CONFIGS:
        m = main[n]
        rows.append([lab(n), f"{m['num_candidate_rules_mean']:.1f}".replace(".", ","), f"{m['num_final_rules_mean']:.1f}".replace(".", ","),
                     f"{m['avg_antecedent_length_mean']:.2f}".replace(".", ","), pct(m["default_usage_rate_mean"]), f"{m['total_seconds']:.2f}".replace(".", ",")])
    out.append(table(f"Tamanho do classificador e custo (seed {MAIN_SEED}; médias por dobra; tempo total das 5 dobras em s).", "tab:tamanho",
                     ["Configuração", "Regras cand.", "Regras finais", "Compr. médio", "Uso do padrão", "Tempo (s)"], rows, "lrrrrr"))

    # 11) por classe (Iris e Pokémon)
    for n in ("iris_disc", "pokemon_disc_group"):
        pc = main[n]["per_class"]
        rows = [[esc(c), num(v["precision"]), num(v["recall"]), num(v["f1"]), str(v["support"])] for c, v in pc.items()]
        out.append(table(f"Métricas por classe (OOF) -- {lab(n)}, seed {MAIN_SEED}.", f"tab:classe_{n}", ["Classe", "Precisão", "Recall", "F1", "Suporte"], rows, "lrrrr"))

    out.extend(build_pyarc_tables())
    out.extend(build_fidelidade_tables())
    return "\n".join(out)


def build_fidelidade_tables() -> list[str]:
    """Tabelas da aproximação às condições do artigo (lê ``fidelidade_paper.json``; ver ``fidelidade_paper.py``)."""
    path = OUT_DIR / "fidelidade_paper.json"
    if not path.exists():
        raise FileNotFoundError("rode antes: python -m experimentos.fidelidade_paper")
    data = json.loads(path.read_text(encoding="utf-8"))
    paper = data["paper_iris"]
    pm = lambda a, f=pct: f"{f(a['mean'])} $\\pm$ {f(a['std'])}"  # noqa: E731
    d1 = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731

    rows = []
    for name, v in data["iris"].items():
        a = v["agg"]
        rows.append([v["label"], pm(a["error"]), f"{pct(a['error']['min'], 1)}--{pct(a['error']['max'], 1)}", d1(a["cars"]["mean"]),
                     d1(a["rules_in_c"]["mean"]), num(a["avg_antecedent_length"]["mean"], 2)])
    rows.append([r"\textbf{Artigo (Tabela 1, CBA CARs)}", pct(paper["error"], 1), "--",
                 f"{paper['cars']} (sem poda) / {paper['prcars']} (com poda)", str(paper["rules_in_c"]), "--"])
    t_iris = table(
        f"Iris, 10 dobras, seeds {data['seeds'][0]}--{data['seeds'][-1]}: da baseline às condições do artigo. Erro = 1 $-$ acurácia OOF (média $\\pm$ desvio entre seeds). "
        "Regras candidatas = CARs (sem poda) ou prCARs (com poda), médias por dobra. ``Artigo'' = valores publicados, de outra implementação e outra partição das dobras.",
        "tab:fid_iris", ["Configuração", "Erro", "Mín--máx", "Regras candidatas", "Regras no classif.", "Compr. médio"], rows, "p{5.6cm}rrrrr")

    rows = []
    for name, v in data["pokemon"].items():
        a = v["agg"]
        rows.append([v["label"], pm(a["accuracy"]), num(a["balanced_accuracy"]["mean"]), num(a["mcc"]["mean"]),
                     pm(a["recall_positive"]), d1(a["cars"]["mean"]), d1(a["rules_in_c"]["mean"])])
    t_pok = table(
        f"Pokémon, divisão por espécie, 5 dobras, seeds {data['seeds'][0]}--{data['seeds'][-1]}: checagem de sensibilidade às condições do artigo (não há número publicado para comparar). Métricas OOF; médias entre seeds.",
        "tab:fid_pokemon", ["Configuração", "Acurácia", "Acc. bal.", "MCC", "Recall $+$", "Regras cand.", "Regras no classif."], rows, "p{4.6cm}rrrrrr")

    cuts = data["iris_full_mdl_cuts"]
    rows = [[esc(c), ", ".join(f"{e:g}".replace(".", ",") for e in v)] for c, v in cuts.items()]
    t_cuts = table("Cortes da discretização por entropia/MDL (Fayyad e Irani) no Iris completo (150 amostras), produzidos por \\texttt{MDLDiscretizer}.",
                   "tab:fid_cortes", ["Atributo", "Cortes"], rows, "lr")
    return [t_iris, t_pok, t_cuts]


PYARC_LABELS = {
    "iris_disc": "Iris discretizado",
    "pokemon_disc_group": "Pokémon discretizado, por espécie",
    "pokemon_disc_row": "Pokémon discretizado, por linha",
}


def build_pyarc_tables() -> list[str]:
    """Tabelas da comparação com o pyARC (lê ``comparacao_pyarc.json``; ver ``comparar_pyarc.py``)."""
    path = OUT_DIR / "comparacao_pyarc.json"
    if not path.exists():
        raise FileNotFoundError("rode antes: python -m experimentos.comparar_pyarc  (precisa de Docker)")
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    frac = lambda a, b: f"{a}/{b}"  # noqa: E731
    rows1, rows2, rows3 = [], [], []
    for cfg, s in summary.items():
        n, v0, v3 = s["folds"], s["v0"], s["v3"]
        rows1.append([PYARC_LABELS[cfg], str(n), frac(s["miner_identical"], n), frac(s["pr_identical"], n),
                      frac(v3["classifier_identical_folds"], n), pct(v3["pred_agreement"])])
        not_ident = n - v0["classifier_identical_folds"]
        rows2.append([PYARC_LABELS[cfg], frac(v0["classifier_identical_folds"], n), str(v0["differs_only_in_tie_order_folds"]),
                      str(not_ident - v0["differs_only_in_tie_order_folds"]), pct(v0["pred_agreement"], 3),
                      pct(v0["metrics"]["accuracy"]), pct(s["ours_metrics"]["accuracy"]), num(v0["metrics"]["mcc"]), num(s["ours_metrics"]["mcc"])])
        rows3.append([PYARC_LABELS[cfg], str(s["ruleitems_ours_total"]), str(s["ruleitems_pyarc_total"]), str(s["empty_antecedent_rules_total"]),
                      str(s["tied_conf_groups"]), str(s["tied_conf_groups_with_float_noise"])])
    seeds = ", ".join(str(x) for x in json.loads(path.read_text(encoding="utf-8"))["seeds"])
    return [
        table(f"Equivalência com o pyARC por nível, em dobras idênticas (seeds {seeds}; 5 dobras por seed). ``M1 isolado'': as nossas CARs alimentando o M1 do pyARC.",
              "tab:pyarc_niveis", ["Configuração", "Dobras", "Mineração idêntica", "Regra possível idêntica", "M1 isolado: classif. idêntico", "M1 isolado: predições iguais"], rows1, "p{3.6cm}rp{1.8cm}p{1.8cm}p{2.2cm}p{2.1cm}"),
        table("pyARC de ponta a ponta (minerador pyfim + M1) versus a nossa baseline. Dobras não idênticas: ordem de regras empatadas (coluna 3) ou conjunto de regras diferente por consequência disso (coluna 4). Métricas OOF agregadas.",
              "tab:pyarc_ponta", ["Configuração", "Classif. idêntico", "Só ordem de empates", "Conjunto diferente", "Predições iguais", "Acur. pyARC", "Acur. nossa", "MCC pyARC", "MCC nosso"], rows2, "p{2.9cm}rp{1.3cm}p{1.5cm}p{1.6cm}rrrr"),
        table("Contagens da comparação (soma das dobras): ruleitems que passam em minsup/minconf (antes da escolha por antecedente), regras de antecedente vazio geradas pelo pyARC, e grupos de regras com mesma confiança exata (empates), dos quais com confianças em ponto flutuante diferentes no pyARC.",
              "tab:pyarc_contagens", ["Configuração", "Ruleitems (nossos)", "Ruleitems (pyARC)", "Antecedente vazio (pyARC)", "Grupos de empate exato", "Empates com ruído de \\emph{float}"], rows3, "p{3.4cm}rrp{1.9cm}p{1.7cm}p{2.2cm}"),
    ]


TEMPLATE = REPO_ROOT / "experimentos" / "relatorio_template.tex"


def montar_relatorio(tables: str) -> Path:
    """Gera ``docs/relatorio_baseline/relatorio_baseline.tex``: UM arquivo autocontido.

    Lê o modelo (``experimentos/relatorio_template.tex``, com marcadores
    ``\\input{tabelas/<rótulo>}``) e substitui cada marcador pela tabela gerada,
    de modo que o resultado pode ser colado inteiro no Overleaf.
    """
    import re

    begin = "\\begin{table}"
    by_label = {}
    for block in tables.split(begin)[1:]:
        label = re.search(r"\\label\{tab:([^}]+)\}", block).group(1)
        by_label[label] = begin + block
    template = TEMPLATE.read_text(encoding="utf-8")
    missing = [m for m in re.findall(r"\\input\{tabelas/([^}]+)\}", template) if m not in by_label]
    if missing:
        raise KeyError(f"tabelas ausentes para o modelo: {missing}")
    full = re.sub(r"\\input\{tabelas/([^}]+)\}", lambda m: by_label[m.group(1)], template)
    path = OUT_DIR / "relatorio_baseline.tex"
    path.write_text(full, encoding="utf-8")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--com-antigo", action="store_true", help=f"também roda o código do commit {OLD_COMMIT}")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data = run_all()
    old = None
    if args.com_antigo:
        print("rodando código antigo...", flush=True)
        old = run_old()
        data["old"] = old
    (OUT_DIR / "dados_baseline.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    montar_relatorio(build_tables(data, old))
    print(f"pronto em {time.perf_counter() - start:.1f}s -> {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
