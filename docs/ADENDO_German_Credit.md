# Adendo — Baseline do CBA num dataset difícil de atributos mistos (German Credit)

Complemento ao relatório da baseline do CBA clássico (branch `Baseline`) e ao `ADENDO_Mushroom.md`. Nenhum código de
`fase2_implementacao/` nem o `runner.py` foi alterado: só foram adicionados dataset, config e teste.

## 1. Motivação e mudança de dataset

O objetivo é mostrar a baseline operando num dataset **genuinamente difícil**, com sobreposição real entre as classes,
complementando o Mushroom (caso trivial, 100%) e o Pokémon (caso real do projeto, só contínuo e desbalanceado).

O dataset pedido inicialmente era o **Pima Indians Diabetes**, mas ele não pôde ser usado com a licença verificada:
o id 17 da UCI é hoje o *Breast Cancer Wisconsin (Diagnostic)*; o Pima não consta na listagem atual da UCI e o caminho
legado devolve 404; espelhos consultados listam a licença como desconhecida e OpenML/Kaggle não foram acessíveis.
Em acordo com o autor, foi usado o **Statlog (German Credit Data)** da UCI, que tem licença verificável e a mesma
característica buscada (teto de desempenho baixo por ruído real). Consequência: **não há a limpeza de zeros
implausíveis do Pima**, porque o German Credit não tem ausentes nem códigos de "não medido".

## 2. Dataset

UCI Statlog (German Credit Data), id 144, Hans Hofmann (1994), DOI 10.24432/C5NC77, **CC BY 4.0**, conferido na página
da UCI em 2026-10-22. 1000 linhas, 20 atributos (7 numéricos + 13 qualitativos), sem valores ausentes. Classe:
`1` bom pagador = 700 (70%), `2` mau pagador = 300 (30%). Detalhes em `datasets/german_credit/README.md`.

Pontos de atenção: (i) a UCI diz que o dataset exige uma matriz de custos (mau como bom custa 5; bom como mau, 1), que o
CBA da baseline **não** usa; (ii) `personal_status_sex` é um atributo sensível, mantido por fidelidade.

## 3. Configuração (`experimentos/configs/german_credit.yaml`)

| Parâmetro | Valor | Por quê |
|---|---|---|
| atributos | 20: 7 contínuos (os "numerical" do `german.doc`) + 13 categóricos nativos | tipagem da própria UCI |
| discretização | igual-frequência, 3 faixas, só nos 7 contínuos | mesmo esquema de Iris/Pokémon; os de 2–4 valores distintos colapsam para 2 faixas efetivas (A07) |
| `min_support` | 0,02 (~16 casos de treino por dobra) | classe positiva com 30%: regras úteis cobrem subgrupos pequenos, mas abaixo disso explodem regras específicas com pouca evidência |
| `min_confidence` | 0,5 | mesmo do Iris/Pokémon |
| `max_antecedent_length` | 3 | convenção da baseline |
| validação | CV estratificada, 5 dobras, por linha, seed 42 | convenção da baseline |
| classe positiva | `2` (mau pagador) | minoritária e a de erro mais caro |

Limiares escolhidos **a priori** e **não ajustados** olhando o resultado (uma única execução).
Comando: `python -m experimentos.runner --config experimentos/configs/german_credit.yaml`.
Execução: `german_credit_discretizado_row_m1_20261022T012220276200Z_f0a7f3` (commit base `e57d77b`, árvore com alterações
pendentes; SHA-256 do `german_credit.csv` = `8f09f11ab68acb98d6f30ef2c2958286f1fc7b6ccd386765544330d156a3b69b`; 5,3 s).
Artefatos em `experimentos/resultados/` (ignorado pelo git); a linha-resumo foi acrescentada a `resumo.csv`.

## 4. Resultados (OOF, 1000 predições, uma por linha)

Matriz de confusão (linhas = classe real, colunas = prevista):

|  | prev. `1` (bom) | prev. `2` (mau) |
|---|---:|---:|
| **real `1`** (bom, 700) | 611 | 89 |
| **real `2`** (mau, 300) | 186 | 114 |

| Métrica | CBA M1 | Classe majoritária (`1`) |
|---|---:|---:|
| Acurácia | **0,7250** | 0,7000 |
| Acurácia balanceada | 0,6264 | 0,5000 |
| Macro-F1 | 0,6348 | 0,4118 |
| MCC | 0,2881 | 0,0000 |
| Recall `1` / `2` | 0,8729 / 0,3800 | 1,0000 / 0,0000 |
| Precisão `1` / `2` | 0,7666 / 0,5616 | 0,7000 / indefinida |

Acurácia por dobra: 0,735 / 0,745 / 0,690 / 0,710 / 0,745 (desvio 0,022). IC 95% (bootstrap por linha): acurácia
[0,697; 0,754], MCC [0,225; 0,359]. Regras candidatas (média): 12 818,6; finais (média): 124,8; antecedente médio: 2,95.
Classe padrão acionada em 203 casos (20,3%).

Custo pela matriz da UCI (informação complementar; o modelo não otimiza custo): 186×5 + 89×1 = **1019** (1,019 por caso)
contra **1500** (1,5) do classificador que sempre diz "bom".

## 5. Auditoria independente

`python -m experimentos.auditoria verify --run experimentos/resultados/german_credit_discretizado_row_m1_20261022T012220276200Z_f0a7f3`
→ **APROVADO**: 1000 predições OOF (uma por linha); predições reconstruídas só dos cortes e regras salvos idênticas às OOF
(1000/1000); acurácia, acurácia balanceada, macro-F1, MCC e matriz de confusão recalculados batem com o resumo; em cada
dobra o prefixo do M1 é o primeiro de erro total mínimo (117–149 regras, erro total 92–114 em ~800 casos de treino).

## 6. Comparação com a literatura

A faixa de ~70–78% de acurácia costuma ser citada na literatura geral para este dataset; **é uma referência genérica,
não conferida contra um artigo específico nesta sessão, e não é uma comparação direta com uma implementação
determinada** (os protocolos de validação, discretizações e pré-processamentos variam entre trabalhos). O resultado,
**72,5%**, está dentro dessa faixa, na metade inferior.

Leitura honesta do resultado:
- O ganho sobre a classe majoritária é **pequeno**: +2,5 pontos de acurácia, e o IC 95% da acurácia ([69,7%; 75,4%])
  inclui os 70% da majoritária. A acurácia sozinha não separa o modelo da baseline trivial.
- Há **sinal real além da majoritária** nas métricas que descontam o desbalanceamento: MCC 0,29 (IC [0,23; 0,36], não
  inclui 0), acurácia balanceada 0,63 e custo 32% menor que o de "sempre bom".
- O modelo é **conservador com a classe minoritária**: recall de maus pagadores de 38% (precisão 56%), com 186 maus
  classificados como bons. Em parte isso reflete a precedência por confiança, que favorece a classe de 70%.
- O modelo se comporta de forma sensata: não degenera para a classe majoritária (como o Pokémon bruto), não passa de 90%
  (sem sinal de vazamento) e mantém dispersão entre dobras compatível com 1000 linhas.

## 7. O que este teste cobre

Atributos **mistos** (7 discretizados + 13 categóricos nativos) em dado real com sobreposição entre classes, classe
desbalanceada 70/30 e ~12,8 mil regras candidatas por dobra. Junto do Mushroom (só categórico, trivial) e do Pokémon (só
contínuo, desbalanceado), cobre os três formatos de entrada do pipeline. `tests/test_german_credit.py` é um smoke test:
carrega o config real, roda a CV, confere uma predição OOF por linha, acurácia entre a majoritária (70%) e 90%, e que a
classe minoritária é predita alguma vez; sem números exatos.
