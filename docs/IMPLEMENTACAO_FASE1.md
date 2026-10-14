# Fase 1 — o que foi implementado e como rodar

Este documento explica, arquivo por arquivo, o que cada módulo criado na Fase 1 do projeto
(baseline do CBA clássico, ver `docs/PIBIC___João_e_Fábio.pdf`) faz e como executá-lo ou
utilizá-lo. Nenhum arquivo aqui é "autoexecutável" sozinho, exceto `experimentos/runner.py` —
os demais são módulos de biblioteca, usados pelos testes e pelo runner, ou importados a partir
de um script/notebook próprio.

## Visão geral

```
fase2_implementacao/
├── association_rules.py   # suporte, confiança, lift, codificação transacional
├── discretization.py       # discretização por igual-frequência (quantis)
├── apriori.py               # mineração de itemsets frequentes (Apriori)
├── fp_growth.py             # mineração de itemsets frequentes (FP-Growth)
└── cba.py                   # CBA-RG (geração de CARs) e CBA-CB M1 (classificador)

tests/
├── conftest.py                    # datasets artificiais usados por todos os testes
├── test_association_rules.py
├── test_apriori.py
├── test_discretization.py
├── test_fp_growth.py
└── test_cba.py

experimentos/
├── runner.py                # ⭐ o único arquivo que se roda diretamente
├── configs/iris.yaml         # configuração do experimento no Iris
└── resultados/                # saídas geradas a cada execução

requirements.txt              # dependências (pandas, numpy, pyyaml, pytest)
datasets/iris.csv              # dataset usado no experimento
```

---

## 1. `fase2_implementacao/association_rules.py`

**O que faz:** calcula as três métricas fundamentais de uma regra de associação — suporte,
confiança e lift — e converte um DataFrame do pandas em uma lista de "transações" (o formato
que todo o resto do pipeline usa: cada linha vira um `frozenset` de pares `(atributo, valor)`,
incluindo a classe como um item especial).

**Funções principais:**
- `support(itemset, transactions)`
- `confidence(antecedent, consequent, transactions)`
- `lift(antecedent, consequent, transactions)`
- `dataframe_to_transactions(df, class_column)`

**Como rodar/usar:** não é um script — é importado por `apriori.py`, `cba.py` e pelo
`experimentos/runner.py`. Para usar isoladamente (ex.: em um notebook ou no console Python):

```python
from fase2_implementacao.association_rules import support, confidence, lift

transactions = [
    frozenset({("Color", "Red"), ("Label", "A")}),
    frozenset({("Color", "Red"), ("Label", "B")}),
]
support({("Color", "Red")}, transactions)                      # 1.0
confidence({("Color", "Red")}, {("Label", "A")}, transactions)  # 0.5
```

---

## 2. `fase2_implementacao/discretization.py`

**O que faz:** transforma atributos contínuos (como as 4 medidas do Iris, em centímetros) em
faixas categóricas, usando quantis (igual-frequência — ex.: tercis, quartis). É a etapa que
precisa vir antes de montar as transações, porque o CBA só entende pares atributo=valor
categóricos.

**Classe principal:** `EqualFrequencyDiscretizer(n_bins, columns)`, com `.fit(df_treino)`,
`.transform(df)` e `.fit_transform(df)`.

Importante: o `.fit` deve ser chamado **só com os dados de treino** de cada fold da validação
cruzada (é assim que o `runner.py` usa) — nunca com o dataset inteiro, para não vazar
informação do teste para o treino.

**Como rodar/usar:**

```python
import pandas as pd
from fase2_implementacao.discretization import EqualFrequencyDiscretizer

df = pd.DataFrame({"PetalLengthCm": [1.4, 1.4, 4.7, 5.1, 6.0]})
disc = EqualFrequencyDiscretizer(n_bins=3, columns=["PetalLengthCm"])
disc.fit_transform(df)
```

---

## 3. `fase2_implementacao/apriori.py`

**O que faz:** minera todos os itemsets frequentes de uma lista de transações — o algoritmo
Apriori clássico, com poda por *downward closure* (descarta candidatos cujo subconjunto já é
sabidamente infrequente). É a base que o CBA-RG (em `cba.py`) usa para gerar as CARs.

**Função principal:** `apriori_frequent_itemsets(transactions, min_support, max_length=None)`
→ retorna `(frequentes, ordem_de_geração)`.

**Como rodar/usar:**

```python
from fase2_implementacao.apriori import apriori_frequent_itemsets

frequentes, ordem = apriori_frequent_itemsets(transactions, min_support=0.25)
```

---

## 4. `fase2_implementacao/fp_growth.py`

**O que faz:** a mesma tarefa do `apriori.py` (minerar itemsets frequentes), mas com o
algoritmo FP-Growth — constrói uma árvore compacta (FP-Tree) em vez de gerar candidatos
explicitamente. Existe para permitir, em fases futuras do projeto, comparar o custo
computacional dos dois algoritmos sobre o mesmo pipeline. **Não é usado pelo CBA-RG atual**
(que usa o Apriori) — é um módulo independente.

**Função principal:** `fp_growth_frequent_itemsets(transactions, min_support, max_length=None)`
→ retorna um dicionário itemset → suporte (mesmo resultado que o Apriori produziria).

**Como rodar/usar:** igual ao `apriori.py`, trocando a importação:

```python
from fase2_implementacao.fp_growth import fp_growth_frequent_itemsets

frequentes = fp_growth_frequent_itemsets(transactions, min_support=0.25)
```

---

## 5. `fase2_implementacao/cba.py`

**O que faz:** as duas etapas centrais do CBA.

- **CBA-RG** (`generate_cars`): usa o Apriori para minerar itemsets e filtra apenas os que têm
  exatamente um item de classe, virando regras `antecedente ⇒ classe`. **Para cada antecedente
  mantém um único consequente** — o de maior confiança (empate: ordem alfabética do rótulo, em
  vez do sorteio do artigo) — e só o aceita se atingir a confiança mínima. Contagens são
  inteiras e os limiares/precedência usam frações exatas (sem comparação por `float`).
- **Ordenação** (`sort_cars_by_precedence`): ordena as CARs por confiança, depois suporte,
  depois ordem de geração — o critério clássico de Liu, Hsu & Ma (1998). É passada como
  parâmetro para o CBA-CB de propósito, para no futuro trocar só esse critério (CBA-QS) sem
  reescrever o resto.
- **CBA-CB, variante M1** (`build_classifier_m1`, Figura 2 do artigo): cobre o conjunto de
  treino sequencialmente com as regras ordenadas e, a cada regra aceita, registra os erros
  acumulados das regras + os erros da classe padrão (maioria dos casos residuais). **Ao final
  trunca a lista no primeiro prefixo de menor erro total** e usa a classe padrão desse prefixo.
  O histórico por prefixo fica em `Classifier.m1_history`. Desvios documentados (desempate
  determinístico, fallback quando tudo é coberto, prefixo vazio fora do corte) estão no
  docstring de `cba.py`. Não há poda pessimista de condições.

**Como rodar/usar:**

```python
from fase2_implementacao.cba import generate_cars, build_classifier_m1

cars = generate_cars(transactions, class_attribute="Label", min_support=0.2, min_confidence=0.5)
classificador = build_classifier_m1(cars, transactions, labels, class_attribute="Label")
classificador.predict(nova_transacao)
```

---

## 6. `tests/` — testes unitários

**O que faz:** verifica cada peça acima contra datasets artificiais pequenos, cujos valores
(suporte, confiança, regras geradas, classe prevista) foram calculados manualmente e estão
documentados nos comentários de cada arquivo de teste. `conftest.py` define os dois datasets
usados por todos os testes (`cores_formas_transactions` e `cba_transactions`).

**Como rodar:**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

(ou apenas `pytest tests/` se o ambiente virtual já estiver ativado)

---

## 7. `experimentos/runner.py` — o pipeline completo (o arquivo que você roda)

**O que faz:** carrega um dataset e um arquivo de configuração YAML, e roda o pipeline inteiro
com validação cruzada estratificada:

1. Separa treino/teste de cada fold (implementação própria, sem scikit-learn).
2. Ajusta a discretização **só com o treino** de cada fold.
3. Roda CBA-RG + CBA-CB M1 no treino do fold.
4. Avalia a acurácia no teste do fold.
5. Registra: semente, tamanho de treino/teste, tempo de execução, nº de regras candidatas e
   finais, comprimento médio do antecedente, acurácia por fold e versões das dependências.

Se o YAML define `group_column` (Pokémon: `"#"`, a espécie), as dobras são construídas **por
grupo** (nenhuma espécie em treino e teste ao mesmo tempo); `--split row` volta à divisão por
linha, para comparação. Na inferência a classe verdadeira nunca entra na transação.

Cada execução grava, sem sobrescrever, um diretório `experimentos/resultados/<run_id>/`
(não versionado) com `summary.json`, `config.json` (commit, hash do dataset), `environment.txt`,
`partitions.json`, `discretization.json`, `rules.json`, `m1_history.json`,
`oof_predictions.csv` e `metrics.json`, e acrescenta uma linha a
`experimentos/resultados/resumo.csv` (versionado; o cabeçalho de arquivos antigos é migrado).
Métricas: acurácia, acurácia balanceada, macro-F1, MCC, matriz de confusão e baseline da
classe majoritária. Para auditar uma execução só pelos artefatos:

```powershell
python -m experimentos.auditoria verify  --run experimentos/resultados/<run_id>
python -m experimentos.auditoria history --run <dir> --fold 0
python -m experimentos.auditoria trace   --run <dir> --row <row_id>
python -m experimentos.metricas --oof <dir>/oof_predictions.csv
```

**Como rodar (Iris):**

```powershell
.venv\Scripts\python.exe -m experimentos.runner --config experimentos\configs\iris.yaml
```

---

## 8. `experimentos/configs/iris.yaml`

**O que faz:** não é código, é a configuração do experimento acima — dataset (`datasets/iris.csv`),
coluna de classe, colunas contínuas a discretizar, número de faixas (`n_bins`), limiares
`min_support`/`min_confidence`, comprimento máximo do antecedente, número de folds e semente.
Para rodar em outro dataset, basta criar um novo arquivo `experimentos/configs/<nome>.yaml`
seguindo o mesmo formato.

---

## 9. Outros arquivos

- **`requirements.txt`** — dependências mínimas (`pandas`, `numpy`, `pyyaml`, `pytest`).
  Instale com `pip install -r requirements.txt` dentro do venv.
- **`datasets/iris.csv`** — o dataset Iris (150 amostras, 3 classes, 4 atributos contínuos),
  renomeado para minúsculo. `datasets/database.sqlite` foi excluído do git (é redundante com o
  CSV) via `.gitignore`.
- **`.gitignore`** — atualizado para ignorar `database.sqlite` e as saídas individuais de
  `experimentos/resultados/` (mantendo apenas `resumo.csv` versionado).
