# Classificador CBA — PIBIC

Projeto de Iniciação Científica (PIBIC) para o desenvolvimento de um classificador baseado em
regras de associação, inspirado no algoritmo **CBA** (Classification Based on Associations),
proposto por Liu, Hsu & Ma (1998). Orientação: Prof. Jonas.

## Status atual

Fase 1 do plano de progressão (ver `docs/PIBIC___João_e_Fábio.pdf`) concluída: CBA clássico
(CBA-RG + CBA-CB M1) reproduzido do zero em `fase2_implementacao/`, com testes unitários
auditáveis à mão em `tests/` e um ambiente de experimentos reprodutível em `experimentos/`,
rodando ponta a ponta no Iris com validação cruzada estratificada. Reuniões de orientação
semanais, às quintas-feiras.

## Roteiro de estudos

### 1. Regras de associação ✅
- [x] O que é uma regra de associação (X → Y, itemsets, X e Y disjuntos)
- [x] Suporte, Confiança e Lift
- [x] Complexidade computacional e itemsets frequentes (2ⁿ, propriedade Apriori / downward closure)
- [x] Discretização de variáveis contínuas (terciles, RMEP)

### 2. Apriori 🔄
- [x] Geração de candidatos por tamanho
- [x] Poda pela propriedade Apriori (downward closure)
- [x] Cálculo de suporte de cada candidato
- [x] Múltiplas varreduras da base de dados
- [x] Geração de regras a partir dos itemsets frequentes
- [x] Filtro final por confiança mínima (minconf)

### 3. FP-Growth 🔄
- [x] Limitações do Apriori que motivam o FP-Growth
- [x] Estrutura FP-Tree
- [x] Ordenação dos itens por frequência
- [x] Construção da FP-Tree
- [x] Conditional pattern base e conditional FP-Tree
- [x] Mineração sem geração explícita de candidatos
- [x] Comparação de eficiência: Apriori vs FP-Growth

### 4. CBA (Classification Based on Associations) ✅
- [x] CARs (Class Association Rules)
- [x] CBA-RG (Rule Generation)
- [x] CBA-CB (Classifier Building) — visão geral
- [x] Poda de regras
- [x] Ordenação das regras
- [x] Database coverage
- [x] Regra padrão (default rule)
- [x] Algoritmo M1 — implementado em `fase2_implementacao/cba.py`, com testes unitários
- [ ] Algoritmo M2 — fica para fase posterior do plano
- [ ] QCBA (Quantitative CBA) — fica para fase posterior do plano

### 5. Plano de progressão (5 fases) ✅ Fase 1 concluída
O relatório de revisão em `docs/PIBIC___João_e_Fábio.pdf` define 5 fases: (1) baseline
auditável — **concluída**; (2) estudo controlado de discretização/limiares; (3) CBA-QS
(seleção de regras por qualidade + interessância + simplicidade); (4) avaliação comparativa;
(5) consolidação. As fases 2–5 ainda não foram iniciadas.

## Metodologia — duas fases

O projeto é dividido deliberadamente em duas fases, para manter clara a autoria no relatório final:

- **`fase1_exploracao/`** — código explorado/adaptado a partir de
  [jirifilip/pyARC](https://github.com/jirifilip/pyARC) (MIT License; Filip & Kliegr, 2018),
  usado exclusivamente para aprendizado prático sobre a implementação do CBA. **Não é
  contribuição original do bolsista.**
- **`fase2_implementacao/`** — implementação própria do zero, construída de forma independente.
  É a única parte que conta como contribuição original para o relatório do PIBIC.

## Estrutura do repositório

```
├── docs/                     # relatório de revisão e plano de progressão do PIBIC
├── fase1_exploracao/         # exploração de código de terceiros (aprendizado)
├── fase2_implementacao/      # implementação própria
│   ├── association_rules.py     # suporte, confiança, lift, codificação transacional
│   ├── discretization.py        # discretização por igual-frequência (quantis)
│   ├── apriori.py                # mineração via Apriori
│   ├── fp_growth.py              # mineração via FP-Growth
│   └── cba.py                    # CBA-RG e CBA-CB (M1)
├── experimentos/             # runner do pipeline completo + configs + resultados
│   ├── runner.py
│   ├── configs/iris.yaml
│   └── resultados/               # saídas por execução (versionado só o resumo.csv)
├── datasets/                 # datasets de teste (ex: iris.csv)
├── notebooks/                 # exploração e visualização
├── tests/                     # testes unitários (base artificial, conferida à mão)
└── CLAUDE.md                  # contexto do projeto para Claude Code
```

## Ambiente técnico

- Python 3.13 (ambiente virtual em `.venv/`)
- Dependências em `requirements.txt` (pandas, numpy, pyyaml, pytest) — instale com
  `pip install -r requirements.txt`

## Rodando

```
pytest tests/
python -m experimentos.runner --config experimentos/configs/iris.yaml
```

## Referências e leituras complementares

### Artigo original do CBA
- Liu, B., Hsu, W., & Ma, Y. (1998). *Integrating Classification and Association Rule Mining*.
  Proc. of the 4th International Conference on Knowledge Discovery and Data Mining (KDD-98),
  80–86. AAAI Press. [DOI](https://dl.acm.org/doi/10.5555/3000292.3000305)

### Extensões e melhorias do CBA
- Liu, B., Ma, Y., & Wong, C.K. (2001). *Classification Using Association Rules: Weaknesses and
  Enhancements*. Propõe o CBA2, corrigindo fraquezas do CBA original.
  [PDF](https://sci2s.ugr.es/keel/pdf/algorithm/capitulo/2001-Liu-CBA2.pdf)
- Kliegr, T. (2017). *Quantitative CBA: Small and Comprehensible Association Rule Classification
  Models*. arXiv:1711.10166. [Link](https://arxiv.org/abs/1711.10166)
- Filip, J., & Kliegr, T. (2018). *Classification based on Associations (CBA) — a performance
  analysis*. EasyChair. [Link](https://easychair.org/publications/preprint/5d6G)
- Yin, X., & Han, J. (2003). *CPAR: Classification based on Predictive Association Rules*.
  Algoritmo relacionado, alternativa ao CBA que evita gerar todas as regras exaustivamente.
- Li, W., Han, J., & Pei, J. (2001). *CMAR: Accurate and Efficient Classification Based on Multiple
  Class-Association Rules*. Outro algoritmo relacionado, usa múltiplas regras na classificação.

### Sobre Bing Liu e classificação associativa
- Bing Liu cunhou o termo "associative classification" (classificação associativa) — ver a
  [página da Wikipedia sobre classificadores associativos](https://en.wikipedia.org/wiki/Associative_classifier)
  para uma visão geral acessível do conceito.
- [Página da Wikipedia sobre Bing Liu](https://en.wikipedia.org/wiki/Bing_Liu_(computer_scientist)) —
  contexto sobre o autor e sua atuação em mineração de dados e aprendizado de máquina.
- Liu, B. *Classification by Association Rule Analysis*, capítulo do *Data Mining and Knowledge
  Discovery Handbook* (Springer) — bom resumo do próprio autor sobre o campo que o CBA fundou.
  [Link](https://link.springer.com/rwe/10.1007/978-0-387-39940-9_558)

### Implementações de referência
- [jirifilip/pyARC](https://github.com/jirifilip/pyARC) — implementação em Python usada na fase 1
  deste projeto.
- [liulizhi1996/CBA](https://github.com/liulizhi1996/CBA) — implementação minimalista, boa para
  ler ao lado do artigo original.
- Pacote `arulesCBA` (R), de Michael Hahsler — implementação de referência amplamente citada,
  endossada pelo próprio Bing Liu. [Documentação](https://rdrr.io/cran/arulesCBA/man/CBA.html)
