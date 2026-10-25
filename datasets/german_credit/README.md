# Statlog (German Credit Data) — UCI Machine Learning Repository

- **Origem:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data — criador: Hans Hofmann
  (doado em 16/11/1994). DOI: 10.24432/C5NC77. Baixado de
  https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip em 2026-10-22.
- **Licença:** **CC BY 4.0** (atribuição obrigatória: citar a fonte acima). Conferido na página do UCI em 2026-10-22,
  que diz: "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license."
- **Conteúdo:** 1000 instâncias, 20 atributos (7 numéricos + 13 qualitativos) e classe binária
  (`1` = bom pagador, 700; `2` = mau pagador, 300). A página informa que não há valores ausentes
  (confirmado: `german_credit.csv` não tem nenhum NaN).

## Por que este dataset (e não o Pima)

O pedido original era o Pima Indians Diabetes. O link `archive.ics.uci.edu/dataset/17/...` hoje é o *Breast Cancer
Wisconsin (Diagnostic)*, o Pima não consta na lista atual de 689 datasets da UCI (a API de listagem não o devolve) e o
caminho legado `/ml/machine-learning-databases/pima-indians-diabetes/` responde 404. Os espelhos consultados
(mldata/TU Berlin, datahub) listam a licença como desconhecida; OpenML e Kaggle não puderam ser acessados daqui.
Sem uma página que permita verificar a licença, optou-se por um dataset difícil da UCI com licença verificável.

## Arquivos

| Arquivo | O que é |
|---|---|
| `german.data` | dados brutos da UCI (separados por espaço, sem cabeçalho, classe na 21ª coluna) — inalterado |
| `german.doc` | documentação original (atributos, códigos, matriz de custos) — inalterada |
| `Index_UCI.txt` | `Index` original do zip — inalterado |
| `german_credit.csv` | os mesmos dados com cabeçalho, lido pelo runner |
| `preparar_csv.py` | gera `german_credit.csv` a partir do `.data` (só acrescenta cabeçalho e troca o separador; nenhum valor muda) |

Nomes de coluna seguem a seção "Attribute description" do `german.doc`. A UCI também distribui
`german.data-numeric` (24 atributos numéricos, versão para algoritmos que exigem números); não foi usado nem copiado.

## Decisões de tratamento

1. **Sem limpeza de dados:** não há ausentes, nem códigos de "não medido". Nada é removido ou convertido.
2. **Tipos:** os 7 atributos que o `german.doc` chama de *numerical* (`duration_months`, `credit_amount`,
   `installment_rate`, `residence_since`, `age`, `existing_credits`, `people_liable`) são tratados como contínuos e
   discretizados; os 13 qualitativos (códigos `A11`, `A12`, ...) entram como categorias nativas. Quatro dos numéricos
   têm só 2–4 valores distintos; o discretizador colapsa faixas vazias (A07).
3. **Classe:** mantida com os códigos originais `1`/`2` (sem renomear para bom/mau).
4. **Matriz de custos:** a UCI informa que o dataset "exige" uma matriz de custos (classificar mau como bom custa 5;
   o inverso, 1). O CBA da baseline **não usa custos**; a matriz só é usada, no adendo, para reportar o custo
   médio como informação complementar.
5. **Atributo sensível:** `personal_status_sex` mistura estado civil e sexo. Foi mantido por fidelidade ao dataset,
   mas qualquer uso do classificador para decisões reais de crédito exigiria revisar isso.
