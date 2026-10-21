"""Mostra no terminal a matriz de confusão de uma execução já salva pelo runner.

Só lê os artefatos (metrics.json, config.json, oof_predictions.csv); não roda o pipeline.
As métricas exibidas são as de metrics.json; o oof_predictions.csv é usado apenas para
conferir (via experimentos.metricas) que a matriz salva bate com as predições salvas.

Uso:
    python -m experimentos.mostrar_matriz --run experimentos/resultados/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from experimentos.metricas import compute_metrics, read_oof


def _load(run_dir: Path) -> tuple[dict, dict, Path]:
    for name in ("metrics.json", "config.json", "oof_predictions.csv"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{name} não encontrado em {run_dir} (isso é uma pasta de execução do runner?)")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return metrics, config, run_dir / "oof_predictions.csv"


def _cell_style(fraction: float, diagonal: bool) -> str:
    """Fundo e texto explícitos (não dependem do tema do terminal).

    Diagonal (acertos) em azul, fora da diagonal (erros) em vermelho; a intensidade é a
    fração da linha (classe real) que caiu naquela célula, para que a classe minoritária
    continue visível mesmo quando a majoritária tem 10x mais exemplos.
    """
    light, dark = ((235, 242, 252), (31, 90, 166)) if diagonal else ((253, 240, 238), (192, 57, 43))
    f = max(0.0, min(1.0, fraction))
    r, g, b = (round(lo + (hi - lo) * f) for lo, hi in zip(light, dark))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#000000" if luminance > 150 else "#ffffff"
    return f"bold {fg} on #{r:02x}{g:02x}{b:02x}"


def _confusion_table(cm: dict) -> Table:
    labels, matrix = cm["labels"], cm["matrix"]
    table = Table(
        title="Matriz de confusão (real × prevista)",
        box=box.HEAVY_HEAD, show_lines=True, title_style="bold",
    )
    table.add_column("real \\ prevista", justify="right", style="bold")
    for label in labels:
        table.add_column(label, justify="center", min_width=12, style="bold")
    table.add_column("n real", justify="right", no_wrap=True)
    for i, label in enumerate(labels):
        row_total = sum(matrix[i])
        cells = []
        for j, count in enumerate(matrix[i]):
            fraction = count / row_total if row_total else 0.0
            text = Text(f"\n{count}\n{fraction:.1%}\n", justify="center")
            text.stylize(_cell_style(fraction, diagonal=(i == j)))
            cells.append(text)
        table.add_row(label, *cells, str(row_total))
    return table


def _context(config: dict, run_id: str) -> Table:
    cfg = config["config"]
    split = cfg.get("split", "row")
    split_txt = f"por grupo (coluna '{cfg['group_column']}')" if split == "group" else "por linha"
    disc = f"sim ({cfg['n_bins']} faixas por quantil)" if cfg.get("discretizar") else "não"
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", justify="right")
    table.add_column()
    table.add_row("Dataset", Path(cfg["dataset_path"]).stem)
    table.add_row("Classe", f"{cfg['class_column']} (positiva: {cfg.get('positive_class')})")
    table.add_row("Discretização", disc)
    table.add_row("Divisão (CV)", f"{split_txt}, {cfg['k_folds']} folds, seed {cfg['seed']}")
    table.add_row("CBA", f"{cfg['cb_variant']}, minsup {cfg['min_support']}, minconf {cfg['min_confidence']}")
    table.add_row("Execução", run_id)
    return table


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _metrics_panel(metrics: dict) -> Table:
    oof, base = metrics["oof"], metrics.get("majority_baseline")
    table = Table(box=box.SIMPLE_HEAD, title="Métricas (predições fora-da-amostra)", title_style="bold")
    table.add_column("métrica")
    table.add_column("CBA", justify="right", style="bold")
    if base:
        table.add_column("classe majoritária", justify="right", style="dim")
    rows = [
        ("Acurácia", oof["accuracy"], base and base["accuracy"]),
        ("Acurácia balanceada", oof["balanced_accuracy"], base and base["balanced_accuracy"]),
        ("Macro-F1", oof["macro_f1"], base and base["macro_f1"]),
        ("MCC", oof["mcc"], base and base["mcc"]),
    ]
    for name, value, ref in rows:
        fmt = (lambda v: f"{v:.3f}") if name == "MCC" else _pct
        table.add_row(name, fmt(value), *([fmt(ref)] if base else []))
    for label, stats in oof["per_class"].items():
        ref = base["per_class"].get(label, {}).get("recall") if base else None
        table.add_row(f"Recall [{label}] (n={stats['support']})", _pct(stats["recall"]), *([_pct(ref)] if base else []))
    return table


def render(run_dir: Path, console: Console) -> None:
    metrics, config, oof_path = _load(run_dir)
    oof = metrics["oof"]
    cm = oof["confusion_matrix"]

    # Confere a matriz salva contra as predições salvas, com a mesma função do runner.
    y_true, y_pred = read_oof(oof_path)
    recomputed = compute_metrics(y_true, y_pred, labels=cm["labels"])["confusion_matrix"]["matrix"]
    check = (Text("✔ matriz confere com oof_predictions.csv", style="green") if recomputed == cm["matrix"]
             else Text("✘ ATENÇÃO: matriz de metrics.json difere de oof_predictions.csv", style="bold red"))

    title = f"CBA — {Path(config['config']['dataset_path']).stem} (n = {oof['n']})"
    legend = Text("cor da célula = fração da classe real (azul: acerto, vermelho: erro)", style="dim")
    console.print()
    console.print(Panel(_context(config, run_dir.name), title=title, title_align="left", border_style="cyan"))
    console.print(_confusion_table(cm), justify="center")
    console.print(legend, justify="center")
    console.print(Panel(_metrics_panel(metrics), border_style="cyan"))
    console.print(check, justify="center")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Mostra a matriz de confusão de uma execução salva do runner.")
    parser.add_argument("--run", required=True, type=Path, help="pasta experimentos/resultados/<run_id>")
    args = parser.parse_args(argv)
    try:
        render(args.run, Console())
    except FileNotFoundError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
