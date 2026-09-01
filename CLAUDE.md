# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a PIBIC (Brazilian undergraduate scientific initiation) research project implementing an
association-rule-based classifier inspired by the **CBA** algorithm (Classification Based on
Associations, Liu/Hsu/Ma 1998). It mines frequent itemsets to generate classification rules and
builds a classifier from them.

## Domain terminology

- **Itemset** — a set of items observed together in a transaction.
- **Support** — proportion of transactions containing a given itemset.
- **Confidence** — proportion of transactions with the antecedent (X) that also contain the
  consequent (Y).
- **Lift** — ratio of observed confidence to the confidence expected under independence; lift < 1
  means negative correlation, lift = 1 means independence, lift > 1 means positive correlation.
- **minsup / minconf** — user-defined minimum thresholds for support and confidence, used to filter
  which itemsets/rules are kept.
- **CAR (Class Association Rule)** — an association rule whose consequent is restricted to a class
  label, rather than any itemset.
- **CBA-RG (Rule Generation)** — the step that mines CARs (via Apriori- or FP-Growth-style frequent
  itemset mining, restricted to class-labeled consequents).
- **CBA-CB (Classifier Building)** — the step that prunes and orders CARs, then builds a classifier
  via database coverage. Two variants: **M1** (sequential prune-then-cover) and **M2** (prunes and
  covers simultaneously).

## Repository structure and the fase1/fase2 authorship split

The project is deliberately split into two phases to keep the PIBIC report's authorship claims
clear:

- **`fase1_exploracao/`** — code explored/adapted from [jirifilip/pyARC](https://github.com/jirifilip/pyARC)
  (MIT-licensed; Filip & Kliegr, 2018), used only to learn how CBA works. This is **not the
  bolsista's own work** — never treat code here as something to extend as if it were original, and
  never attribute it to the project author. If anything from this phase influences the final report
  or fase2_implementacao/, cite the original source explicitly.
- **`fase2_implementacao/`** — the actual from-scratch implementation, and the only code that counts
  as the bolsista's contribution for the PIBIC report:
  - `association_rules.py` — support, confidence, and lift calculations
  - `apriori.py` — Apriori frequent-itemset mining
  - `fp_growth.py` — FP-Growth frequent-itemset mining (FP-Tree based alternative to Apriori)
  - `cba.py` — CBA-RG (rule generation) and CBA-CB (classifier building, M1/M2 variants)

When asked to implement or fix the classifier itself, work in `fase2_implementacao/`, not
`fase1_exploracao/`. It is fine to read `fase1_exploracao/` for reference on how pyARC approached a
problem, but the `fase2_implementacao/` code must remain an independent implementation, not a copy.

All `.py` files in `fase2_implementacao/` currently contain only a module-level docstring describing
their purpose — no implementation yet.

Other top-level directories:
- `datasets/` — test datasets (e.g. Iris)
- `notebooks/` — Jupyter notebooks for exploration/visualization
- `tests/` — unit tests for `fase2_implementacao/`

## Environment

- Python 3.13 virtualenv at `.venv/` (created via `virtualenv`, no packages installed yet).
- No `requirements.txt`, `pyproject.toml`, or dependency manifest exists yet — add one when
  introducing the first dependency (e.g. numpy/pandas for data handling, pytest for testing).
- No test runner, linter, or build tooling is configured yet.