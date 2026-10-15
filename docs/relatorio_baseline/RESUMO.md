# Resumo — Baseline do CBA clássico (branch `Baseline`)

Relatório completo: `relatorio_baseline.tex`. Todos os números abaixo vêm de
`python -m experimentos.gerar_relatorio --com-antigo` (dados em `dados_baseline.json`).

## O que foi feito
Correção dos 12 achados (A01–A12) do Relatório 02 do orientador, na ordem dos 5 estágios:

| Estágio | Itens | Essência |
|---|---|---|
| 1 | A01, A02, A10 | M1 com corte de erro mínimo por prefixo (Fig. 2 do paper); um consequente por antecedente; confiança/precedência com frações exatas |
| 2 | A03, A07, A08, A09 | Discretizador (NaN rejeitado, cortes degenerados, base só-categórica); validação de config; `positive_class` normalizada |
| 3 | A04, A05, A06 | Inferência sem a classe; divisão por espécie (`group_column`); estratificação sem folds vazios |
| 4 | A11, A12 | Métricas (acc. balanceada, macro-F1, MCC), predições OOF, regras, cortes, histórico do M1, `run_id`, ambiente |
| 5 | A12 | Apriori com contagem vertical (≈2,7–12×) e FP-Growth ponderado (≈1,1–3×), resultados idênticos |

CBA-QS: **congelado** (não editado nem validado); herda o corte do M1 por reutilizar `build_classifier_m1`.

## Resultados (seed 42, métricas OOF)
| Configuração | Acurácia | Majoritária | MCC | Recall da classe `True` |
|---|---|---|---|---|
| Iris discretizado | 95,33% | 33,33% | 0,931 | — |
| Iris bruto | 78,67% | 33,33% | 0,686 | — |
| Pokémon discretizado, por espécie | 94,13% | 91,88% | 0,598 | 61,54% |
| Pokémon discretizado, por linha | 94,50% | 91,88% | 0,616 | 61,54% |
| Pokémon bruto (espécie ou linha) | 91,88% | 91,88% | 0 | 0% |

Em 10 seeds (média ± desvio): Iris disc. 93,80% ± 0,85; Pokémon disc. por espécie
93,33% ± 0,42 (MCC 0,555; recall 59,2%), por linha 94,00% ± 0,51 (MCC 0,601; recall 63,7%).

## Antes × depois (mesma divisão por linha)
- Pokémon discretizado: 92,00% → 94,50%; recall `True` 1,54% → 61,54%
  (TN/FP/FN/TP: 735/0/64/1 → 716/19/25/40, igual ao medido pelo orientador).
- Iris discretizado: 94,00% → 95,33% na seed 42, mas só 93,67% → 93,80% em 10 seeds
  (o ganho do Iris é em boa parte efeito da seed).
- Sem discretização: sem mudança relevante (Pokémon continua = classe majoritária).

## Conclusões
1. O corte do M1 (A01) explica a grande mudança no Pokémon: antes, a classe padrão era sempre
   `False`; com o corte, o M1 para antes de cobrir tudo e a padrão passa a `True`.
2. Acurácia isolada engana no Pokémon (majoritária = 91,88%); usar MCC, acurácia balanceada e recall.
3. Separar espécies reduz o desempenho de forma sistemática, porém pequena (acurácia −0,67 pp em 8/10
   seeds; MCC −0,046 em 10/10): na seed 42 isolada o efeito parecia ruído — só a varredura revelou.
4. Sem discretização o classificador degenera (Pokémon: 63 regras candidatas, todas `False`).
5. Incerteza grande na classe minoritária (65 positivos): IC 95% do MCC no Pokémon [0,45; 0,72].

## Verificação
- 463 testes passam (dos 36 originais, 35 intactos e 1 corrigido, pois fixava o M1 sem corte).
- `auditoria verify` (aprovado): 800/800 e 150/150 predições reconstruídas só dos artefatos salvos;
  métricas recalculadas das predições OOF idênticas ao resumo; prefixo do M1 = primeiro mínimo em cada dobra.

## Não implementado / limitações
Categoria "ausente" para NaN (hoje rejeita); FP-Growth no CBA-RG; seleção de hiperparâmetros e
AUC (não se aplicam); M2. (A poda pessimista existe agora como opção, ver abaixo.) Desvios do paper documentados: desempate determinístico,
fallback de classe padrão, prefixo vazio fora do corte, discretização por quantis.

## Comparação com o pyARC (referência externa)
pyARC 1.1.4 (MIT) rodado em Docker (`experimentos/comparar_pyarc.py`), nas MESMAS dobras e dados
discretizados: 90 dobras = Iris, Pokémon por espécie, Pokémon por linha × seeds 42, 0–4 × 5 dobras.

| Nível | Resultado |
|---|---|
| Mineração (ruleitems e contagens) | idêntica em 90/90 dobras |
| Regra possível (1 consequente/antecedente) | idêntica em 90/90 |
| CBA-CB isolado (nossas CARs no M1 do pyARC) | classificador e predições idênticos em 90/90 |
| Ponta a ponta (pyARC puro) | predições iguais em 100% (Iris) e 99,98% (Pokémon: 1 caso em 4.800) |

- A diferença ponta a ponta vem só da **ordem entre regras empatadas** (pyARC: ordem do pyfim; aqui: ordem do Apriori).
- Nestes dados, nem a PR (A02), nem as regras de antecedente vazio do pyARC, nem o ruído de float (A10) mudaram o resultado:
  v0 = v1 = v2 em 90/90. A02 e A10 são correções de fidelidade, sem efeito mensurável aqui.
- Cuidado de uso do pyARC: com `unique_transactions=False` (padrão), o M1 colapsa duplicatas e erra.
- Limites: só M1; mesma discretização e dobras; as duas implementações partem do mesmo artigo.

## Fidelidade ao artigo original (opções novas, desligadas por padrão)
Adicionadas sem alterar nenhum resultado da baseline (conferido: Iris 95,33%; Pokémon 94,13% por espécie / 94,50% por linha; bruto 91,88%):
- **Discretização por entropia/MDL** (Fayyad e Irani, a do artigo): `discretization_method: mdl`. Validada com cálculos à mão, contra
  uma implementação ingênua em 60 casos e no Iris completo (cortes 5,55/6,15; 2,95/3,35; 2,45/4,75; 0,8/1,75, os clássicos desse dataset;
  lembrados de memória, não confrontados com o Weka).
- **Poda pessimista de condições** (opcional no artigo): `prune: true`, nível de confiança 0,25. A fórmula do erro pessimista não está
  no artigo: usamos o limite binomial superior exato (escolha nossa, documentada).

**Iris nas condições do artigo** (MDL, minsup 1%, sem limite de antecedente, 10 dobras; 10 seeds), config `iris_paper.yaml`:

| | Erro | CARs (sem poda) | prCARs (com poda) | Regras no classificador |
|---|---|---|---|---|
| Artigo (Tabela 1) | 7,1% | 72 | 23 | 5 |
| Esta baseline | 6,53% ± 0,88 | 92,3 | 21,3 | 4,7 |
| Baseline atual (quantis, 5%, antec. ≤ 3) | 6,47% ± 0,79 | 77,0 | — | 8,3 |

- Erro e tamanho do classificador são compatíveis (diferença de 0,57 pp < 1 desvio e < 1 caso em 150). **Número de CARs sem poda: 28% acima do
  artigo, sem causa identificada** (não temos o código original). Com poda, 21 vs 23.
- A poda corta ~77% das CARs sem mudar o erro (de acordo com o artigo).
- "Mais perto do artigo" não é "melhor": quantis com 1% e sem limite deram 4,07% de erro.
- Pokémon (só sensibilidade; sem número publicado): conclusões mantidas (recall ~56–62%, MCC 0,55–0,59); MDL reduz o classificador de 42 para ~21 regras.
- Implicação para o CBA-QS: a poda se sobrepõe conceitualmente à penalização de comprimento; comparar contra CBA **e** CBA com poda.
