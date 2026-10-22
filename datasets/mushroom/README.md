# Mushroom (UCI Machine Learning Repository)

- **Origem:** https://archive.ics.uci.edu/dataset/73/mushroom — doado por Jeff Schlimmer (27/04/1987);
  registros extraídos de *The Audubon Society Field Guide to North American Mushrooms* (G. H. Lincoff, 1981).
  DOI: 10.24432/C5959T. Baixado de https://archive.ics.uci.edu/static/public/73/mushroom.zip em 2026-10-21.
- **Licença:** **CC BY 4.0** (atribuição obrigatória: citar a fonte acima). Conferido na página do UCI em 2026-10-22,
  que diz: "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license."
  (https://creativecommons.org/licenses/by/4.0/legalcode).
- **Conteúdo:** 8124 instâncias, 22 atributos categóricos + classe binária (`e` comestível = 4208, `p` venenoso = 3916).

## Arquivos

| Arquivo | O que é |
|---|---|
| `agaricus-lepiota.data` | dados brutos da UCI (sem cabeçalho, classe na 1ª coluna) — inalterado |
| `agaricus-lepiota.names` | documentação original (atributos, valores, classes) — inalterado |
| `README_UCI.txt` | README original do arquivo UCI — inalterado |
| `mushroom.csv` | os mesmos dados com cabeçalho, lido pelo runner |
| `preparar_csv.py` | gera `mushroom.csv` a partir do `.data` (só acrescenta o cabeçalho; nenhum valor muda) |

Nomes de coluna seguem `agaricus-lepiota.names`; a única diferença é `bruises?` → `bruises`.

## Decisões de tratamento

1. **Valores ausentes em `stalk-root` (2480 linhas, 30,5%; todos marcados `?`).**
   **Decisão: manter `?` como uma categoria própria** (`stalk-root=?`), sem remover linhas nem a coluna.
   Motivos: (a) preserva as 8124 linhas e a comparabilidade com a literatura; (b) a ausência aqui não é
   aleatória — é uma propriedade do registro (raiz não observável) e pode ser informativa, e o CBA, que só
   enxerga itens `(coluna, valor)`, a trata naturalmente como mais um valor; (c) descartar a coluna
   perderia 1 dos 22 atributos só para evitar um caso que não quebra o pipeline. Não aciona a política
   de rejeição de NaN (A03): o pandas lê `?` como string, e `mushroom.csv` não contém nenhum NaN
   (verificado: `df.isna().sum().sum() == 0`). Regras com `stalk-root=?` significam "raiz ausente no
   registro", não um valor a imputar.
2. **`veil-type` é constante** (só `p`) em todas as linhas. Mantido por fidelidade aos 22 atributos originais;
   com suporte de 100%, o item não discrimina classe.
3. **Sem discretização:** nenhum atributo é numérico; cada categoria vira diretamente o item `(coluna, valor)`.
4. **Sem coluna de identificador/grupo:** a divisão é por linha, estratificada pela classe.
