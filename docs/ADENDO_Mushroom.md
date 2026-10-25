# Adendo — Validação da baseline em atributos categóricos nativos (Mushroom)

Complemento ao relatório da baseline do CBA clássico (branch `Baseline`, commit `e57d77b`). Nenhum código
de `fase2_implementacao/` nem o `runner.py` foi alterado: só foram adicionados dataset, config e teste.

## 1. Motivação

Iris e Pokémon têm atributos contínuos, que passam pelo discretizador. O caminho com **atributos categóricos
nativos, sem discretização** (achado A08) só tinha cobertura de teste unitário sintético. Este adendo o
exercita num dataset real, de formato diferente, com o pipeline já validado.

## 2. Dataset

UCI Mushroom (id 73; Schlimmer, 1987; CC BY 4.0, conferido na página do UCI em 2026-10-22): 8124 registros, 22 atributos
categóricos (1 a 12 valores), classe binária: comestível `e` = 4208 (51,8%), venenoso `p` = 3916 (48,2%).
Detalhes e licença em `datasets/mushroom/README.md`.

**Valores ausentes.** `stalk-root` tem 2480 `?` (30,5%). Decisão: **`?` é mantido como categoria própria**
(`stalk-root=?`). Preserva todas as linhas e os 22 atributos; a ausência é uma característica do registro
original, não um NaN do pipeline, portanto a rejeição de NaN (A03) não é acionada (o CSV não contém NaN).
`veil-type` é constante (só `p`) e foi mantido por fidelidade ao conjunto original.

## 3. Configuração (`experimentos/configs/mushroom.yaml`)

| Parâmetro | Valor | Por quê |
|---|---|---|
| atributos | os 22, todos categóricos; `continuous_columns: []` | sem numéricos |
| `discretizar` | `false` | nada a discretizar |
| `min_support` | 0,01 | ~65 linhas de treino por dobra; mantém regras de subgrupos pequenos (ex.: um odor específico) com 117 itens distintos; o 0,05 do Iris cortaria regras úteis |
| `min_confidence` | 0,5 | mesmo do Iris/Pokémon |
| `max_antecedent_length` | 3 | mesma convenção da baseline |
| validação | CV estratificada, 5 dobras, divisão por linha, seed 42 | convenção da baseline; sem coluna de grupo |
| `cb_variant` | `m1` | CBA-CB clássico |

Os limiares foram escolhidos **a priori** e **não foram ajustados** olhando o resultado (uma única execução).
Comando: `python -m experimentos.runner --config experimentos/configs/mushroom.yaml`.
Execução: `mushroom_sem_discretizacao_row_m1_20261022T005829869206Z_2ba62b` (commit `e57d77b`, árvore limpa;
SHA-256 do `mushroom.csv` = `7204d38399e2c7c84d85e7a2019081724bb943613e82be5d3523156cfbcf9f64`; 33 s).
Os artefatos completos ficam em `experimentos/resultados/` (ignorado pelo git, como nas demais execuções);
a linha-resumo foi acrescentada a `resumo.csv`.

## 4. Resultados (OOF, 8124 predições, uma por linha)

Matriz de confusão (linhas = classe real, colunas = prevista):

|  | prev. `e` | prev. `p` |
|---|---:|---:|
| **real `e`** (comestível) | 4208 | 0 |
| **real `p`** (venenoso) | 0 | 3916 |

| Métrica | CBA M1 | Classe majoritária (`e`) |
|---|---:|---:|
| Acurácia | **1,0000** | 0,5180 |
| Acurácia balanceada | 1,0000 | 0,5000 |
| Macro-F1 | 1,0000 | 0,3412 |
| MCC | 1,0000 | 0,0000 |
| Recall `e` / `p` | 1,0000 / 1,0000 | 1,0000 / 0,0000 |

IC 95% (bootstrap por linha): acurácia e MCC [1,0000; 1,0000]; desvio entre dobras = 0 (as 5 dobras acertaram 100%).
Regras candidatas (média): 27 190,8; regras finais (média): 36,6 (36–37 por dobra); antecedente médio: 2,58.
Classe padrão acionada em 224 casos (2,76%), todos corretos.

## 5. Auditoria independente

`python -m experimentos.auditoria verify --run experimentos/resultados/mushroom_sem_discretizacao_row_m1_20261022T005829869206Z_2ba62b`
→ **APROVADO**: 8124 predições OOF (uma por linha); predições reconstruídas só a partir dos cortes e regras
salvos idênticas às OOF (8124/8124); acurácia, acurácia balanceada, macro-F1, MCC e matriz de confusão
recalculados batem com o resumo; em cada dobra o prefixo do M1 escolhido é o primeiro de erro total mínimo.

## 6. Interpretação e cuidados

- O resultado perfeito **não indica vazamento**: o Mushroom é quase determinístico em seus atributos (é comum
  na literatura obter ~100% com classificadores de regras), o pipeline não usa a classe na inferência (A04) e
  as regras do M1 são legíveis (ex.: `odor=f ⇒ p`; `gill-size=b ∧ odor=n ∧ ring-number=o ⇒ e`).
- Por isso o dataset tem **efeito-teto**: serve como evidência de **funcionamento correto** do caminho
  categórico, **não** para discriminar o CBA de outros métodos nem para estimar desempenho em dados difíceis
  (Iris e Pokémon seguem como referência para isso).
- Não foi verificada equivalência numérica com outra implementação (ex.: pyARC) neste dataset.

## 7. O que este teste cobre

Atributos **categóricos nativos, sem discretização**, em dado real (8124 linhas, 117 itens distintos, ~27 mil
regras candidatas por dobra), incluindo uma categoria `?` tratada como valor. Antes disso, esse caminho só tinha
cobertura de teste unitário sintético (`tests/test_runner.py::test_categorical_only_dataset_runs_with_discretizar_true`).
Agora há também `tests/test_mushroom.py` (smoke test: carrega o config real, roda a CV, confere uma predição OOF
por linha e acurácia acima da classe majoritária, sem números exatos).
