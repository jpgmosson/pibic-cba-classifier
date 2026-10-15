"""Lado do pyARC da comparação (roda DENTRO do container Docker, ver comparar_pyarc.py).

Lê ``/work/input.json`` (dobras já discretizadas pelo nosso pipeline, mais as CARs
da nossa implementação) e escreve ``/work/pyarc.json`` com o que o pyARC
(https://github.com/jirifilip/pyARC, MIT) produz nas MESMAS dobras:

- ``v0``: pyARC de ponta a ponta (``CBA(...).fit``): mineração própria (pyfim) + M1.
- ``v1``: CARs do pyARC sem as regras de antecedente vazio, depois M1 do pyARC.
- ``v2``: v1 + regra possível (PR) do artigo (um consequente por antecedente), depois M1 do pyARC.
- ``v3``: as CARs da NOSSA implementação alimentando o M1 do pyARC (isola o CBA-CB).

Este arquivo não faz parte da implementação (``fase2_implementacao``): é só instrumentação
para usar o pyARC como referência externa. Nenhum código do pyARC foi copiado para o projeto.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from pyarc.algorithms import M1Algorithm, createCARs, generateCARs
from pyarc.data_structures import TransactionDB

SEP = ":=:"


def car_tuple(car):
    ant = sorted(f"{a}{SEP}{v}" for a, v in car.antecedent.itemset.items())
    return (f"{car.consequent.attribute}{SEP}{car.consequent.value}", ant, car.support, car.confidence)


def describe(car):
    con, ant, support, confidence = car_tuple(car)
    return {"ant": ant, "con": con, "support": support, "confidence": confidence, "rid": car.rid}


def run_m1(tuples, train_db, test_db):
    """M1 do pyARC sobre CARs construídas a partir de tuplas (con, ant, sup, conf), na ordem dada (rid)."""
    cars = createCARs(list(tuples))
    clf = M1Algorithm(cars, train_db).build()
    return {
        "rules": [[sorted(f"{a}{SEP}{v}" for a, v in r.antecedent.itemset.items()), f"{r.consequent.attribute}{SEP}{r.consequent.value}"] for r in clf.rules],
        "default": str(clf.default_class),
        "n_cars": len(cars),
        "pred": [str(p) for p in clf.predict_all(test_db)],
    }


def main() -> None:
    data = json.load(open("/work/input.json", encoding="utf-8"))
    out = {}
    for key, case in data["cases"].items():
        header = case["header"]
        train = pd.DataFrame(case["train"], columns=header)
        test = pd.DataFrame(case["test"], columns=header)
        # unique_transactions=True é essencial: com o padrão, set(dataset) do M1 colapsa duplicatas.
        train_db = TransactionDB.from_DataFrame(train, unique_transactions=True)
        test_db = TransactionDB.from_DataFrame(test, unique_transactions=True)
        minsup, minconf, maxlen = case["minsup"], case["minconf"], case["maxlen"]

        # v0: ponta a ponta, exatamente como o README do pyARC recomenda (support/confidence em %, maxlen = nº de itens da regra)
        raw_cars = generateCARs(train_db, support=minsup * 100, confidence=minconf * 100, maxlen=maxlen)
        ordered = sorted(raw_cars, key=lambda c: c.rid)  # ordem de saída do fim, antes do sort por precedência
        raw_tuples = [car_tuple(c) for c in ordered]
        v0 = run_m1(raw_tuples, train_db, test_db)

        # v1: sem antecedente vazio
        v1_tuples = [t for t in raw_tuples if t[1]]
        v1 = run_m1(v1_tuples, train_db, test_db)

        # v2: v1 + PR (um consequente por antecedente: maior confiança; empate -> rótulo alfabético)
        best = {}
        for con, ant, sup, conf in v1_tuples:
            k = tuple(ant)
            if k not in best or (-conf, con) < (-best[k][3], best[k][0]):  # maior confiança; empate: menor rótulo
                best[k] = (con, ant, sup, conf)
        v2_tuples = [t for t in v1_tuples if best[tuple(t[1])] == t]
        v2 = run_m1(v2_tuples, train_db, test_db)

        # v3: nossas CARs -> M1 do pyARC
        ours = [(c["con"], c["ant"], c["support"], c["confidence"]) for c in case["our_cars"]]
        v3 = run_m1(ours, train_db, test_db)

        out[key] = {
            "pyarc_cars": [describe(c) for c in ordered],
            "n_empty_antecedent": sum(1 for t in raw_tuples if not t[1]),
            "v0": v0, "v1": v1, "v2": v2, "v3": v3,
        }
        print("ok", key, file=sys.stderr, flush=True)
    json.dump(out, open("/work/pyarc.json", "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
