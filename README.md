# PIBIC — Classificador Baseado em Regras de Associação (CBA)

Projeto de Iniciação Científica (PIBIC) sobre um classificador baseado em
regras de associação, inspirado no algoritmo **CBA** (Classification Based on
Associations), proposto por Liu, Hsu e Ma (1998).

## Estrutura do projeto

O projeto está dividido em duas fases, para deixar clara a distinção entre
código de terceiros usado para aprendizado e a implementação própria feita
durante a Iniciação Científica:

- **`fase1_exploracao/`** — código explorado e adaptado a partir do
  repositório [jirifilip/pyARC](https://github.com/jirifilip/pyARC), usado
  apenas como referência para entender o funcionamento do CBA antes de
  implementá-lo. **Não é autoria do bolsista.**

- **`fase2_implementacao/`** — implementação própria do CBA, feita do zero,
  incluindo:
  - `association_rules.py`: cálculo de suporte, confiança e lift;
  - `apriori.py`: algoritmo Apriori para mineração de itemsets frequentes;
  - `fp_growth.py`: algoritmo FP-Growth, alternativa mais eficiente ao
    Apriori;
  - `cba.py`: CBA-RG (geração de regras) e CBA-CB (construção do
    classificador, variantes M1/M2).

  **Esta é a implementação que representa a contribuição do bolsista e que
  deve ser considerada no relatório do PIBIC.**

- **`datasets/`** — datasets usados para testar e validar o classificador
  (ex.: Iris).

- **`notebooks/`** — notebooks Jupyter para exploração dos dados,
  experimentos e visualização de resultados.

- **`tests/`** — testes unitários da implementação em `fase2_implementacao/`.
