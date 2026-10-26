# Adendo — Baseline do CBA num dataset pequeno e real de municípios (Municípios de SC)

Complemento ao relatório da baseline do CBA clássico (branch `Baseline`) e aos adendos `ADENDO_Mushroom.md` e
`ADENDO_German_Credit.md`. Nenhum código de `fase2_implementacao/` nem o `runner.py` foi alterado: só foram
adicionados dataset, script de preparação, config e teste.

## 1. Motivação

Terceiro teste de robustez da baseline, agora num dado **fora dos repositórios públicos**, pequeno (292 linhas),
de atributos mistos e com um alvo (`predominance_2022`, Urbano/Rural) que tem **colunas quase determinísticas no próprio
dado**. Isso obriga a decidir o que é atributo legítimo e o que é vazamento, e testa a baseline justamente nessa
fronteira.

## 2. Dataset: proveniência e licença

- **Origem:** arquivo `05_banco_disc_mineracao_final.xlsx`, enviado pelo professor Fábio (disciplina de mineração de
  dados) para fins acadêmicos. **Não é um dataset público tipo UCI**: não há página, DOI nem licença verificável.
- **Redistribuição: autorizada pelo professor, por mensagem.** Em 25/10/2026, pelo WhatsApp, o professor Fábio respondeu:
  "Sim João, libero a distribuição da dataset". É um registro informal: **não há licença formal nem termo de uso**
  (nenhuma licença do tipo CC foi atribuída ao conjunto). As estatísticas municipais de origem (IBGE etc.) são públicas, mas
  a **compilação** é do professor. O `.xlsx` original **não é versionado**; só o CSV limpo é.
- **Conteúdo:** aba `banco` com 295 municípios de Santa Catarina e 42 colunas (a tabela de dados começa na linha 20;
  as linhas 1-18 são um painel de estatísticas pré-calculado, não são dados). A aba `Dicionario` define as variáveis,
  mas descreve colunas que não estão na tabela (`Population_2000/2010`, `gdp_*`, `AREA_KM2_*`) e não descreve outras
  que estão (`dom_tot`, `qnt_pd`, `MP`, `total_instrumentos`, `gap_medio_pd`).
- **Alvo:** `predominance_2022`: Urbano 209, Rural 86 (70,8% / 29,2%) nos 295; após a limpeza, 206 / 86 em 292
  (70,5% / 29,5%).

## 3. Pré-processamento (`datasets/preparar_municipios_sc.py`)

295 × 42 → **292 × 27** (26 atributos + alvo). Cada decisão e sua justificativa:

| Ação | Colunas | Por quê |
|---|---|---|
| Removidas como identificadores | `cities`, `Codigo_IBGE` | não são atributo; um identificador único não generaliza |
| Removidas: **determinam o alvo** | `Pop_urb`, `Pop_rur`, `dens_urb_22`, `dens_rur` | verificado no dado: `predominance_2022 == (Pop_urb / Population_2022 ≥ 0,5)` em 295/295 linhas e `Pop_urb + Pop_rur == Population_2022`; as densidades são `Pop_* / area_*` (`dens_urb_22 × area_urb == Pop_urb`) |
| Removidas: **proxies diretos da fórmula** | `Domicilios_urb`, `Domicilios_rur` | o mesmo recorte urbano/rural, em domicílios (`Domicilios_urb + Domicilios_rur == dom_tot`); não entram na fórmula do alvo, mas a aproximam de perto |
| Removidas: **dependem do alvo** | `change_2010_2022` | é a comparação entre `predominance_2010` e `predominance_2022` |
| Removidas: **proxies, não vazamento por definição** | `area_urb`, `area_rur`, `Urb %`, `Area_Pred` | vêm da mesma classificação censitária de setores urbano/rural de 2022. **Não determinam o alvo**: `Area_Pred` é "Rural" em 288 municípios, 202 deles Urbanos pelo critério populacional. Removidas por precaução (decisão do autor), não por serem triviais |
| Removida: ausentes demais | `gap_medio_pd` | 168 de 295 (57%) ausentes, alto demais para imputar |
| Removida: ausentes e sem explicação | `total_instrumentos` | ver abaixo |
| Linhas descartadas (3) | `Ouro` (sem `IDH_2010`), `Pescaria_Brava` e `Balneario_Rincao` (sem `growth_2000_2010`, `pop_2010_2022`, `growth_%_2010_2022`, `IDH_2010` e `predominance_2010`) | perda de 1% das linhas; as três são Urbano, então a proporção Urbano/Rural passou de 70,8% para 70,5% |
| Mantida como atributo | `predominance_2010` | medição histórica independente; não é derivada de nenhuma coluna de 2022 presente. **Mas é o atributo que domina o resultado (seção 6)** |

**`total_instrumentos` (51 ausentes, 17%): coluna excluída, não imputada.** A decisão foi tomada olhando só a
estrutura dos ausentes, não a relação com o alvo. (i) A ausência não é estrutural: dos 51 ausentes, só 4 têm
`qnt_pd = 0` (contra 34 entre os 244 presentes) e a distribuição de `MP` e `Council_PD` entre ausentes e presentes é
parecida, então nenhuma outra coluna explica o "ausente"; (ii) é uma contagem (0 a 6) que o dicionário não define, e
imputar 17% dela com um valor típico seria inventar o dado; (iii) descartar as 51 linhas custaria 17% de uma base já
pequena; (iv) o CBA da baseline rejeita NaN (A03), então manter a coluna exigiria uma das duas coisas anteriores. O
custo é perder um atributo possivelmente informativo.

**Tipos.** 15 atributos numéricos contínuos (`revenue`, `itc`, `iptu`, `issqn`, `transfer`, `city_birth`,
`growth_2000_2010`, `area_tot`, `dom_tot`, `dens_tot`, `Population_2022`, `pop_2010_2022`, `growth_%_2010_2022`,
`IDH_2010`, `qnt_pd`) e 11 categóricos nativos (`year_MP`, `Council_PD`, `reason_MP`, `predominance_2010`,
`Microrregion`, `rgi`, `Region`, `Metropolitan_Region`, `Association`, `MP`, `hierarchy`). `year_MP` mistura anos com o
valor `non_located` (39 municípios) e entra como categórica de 33 valores; `IDH_2010` está em milésimos (ex.: 704),
não de 0 a 1 como diz o dicionário. Nenhum valor das colunas mantidas foi alterado ou imputado.

## 4. Discretização: esquema do professor ou rotina do projeto?

**Prevalece a rotina do projeto** (igual-frequência, 3 faixas, ajustada **só nos dados de treino de cada dobra**), pelos
motivos:

1. **É o mesmo método.** O painel das linhas 1-8 da aba `banco` são os tercis (1/3 e 2/3) de cada coluna. Conferi os
   cortes do painel contra `np.quantile`, que é o que o discretizador do projeto usa: **25 das 26 colunas numéricas do
   painel coincidem exatamente**. A exceção é `IDH_2010` (painel: 714 e 749; recalculado a partir da tabela: 705 e
   745), cuja causa não foi investigada. Há consistência de método, e não há ganho em copiar os números.
2. **O painel foi calculado com os 295 municípios**, inclusive os que ficam no teste de cada dobra. Reaproveitá-lo faria a
   discretização "ver" os dados de teste (o corte depende deles, mesmo sem usar o alvo), o que o contrato do
   pipeline evita ao ajustar os cortes só no treino.
3. O painel cobre colunas que foram removidas e tem linhas arredondadas (38 000 000 em vez de 38 353 056,67), que não
   são reprodutíveis pelo pipeline.

Resultado: nas 5 dobras, as 15 colunas ficaram com **3 faixas efetivas** (nenhum colapso por empates, A07).

## 5. Configuração (`experimentos/configs/municipios_sc.yaml`)

| Parâmetro | Valor | Por quê |
|---|---|---|
| atributos | 26: 15 contínuos discretizados + 11 categóricos nativos | seção 3 |
| `discretizar` / `n_bins` | `true` / 3 | mesmo esquema de Iris, Pokémon e German Credit |
| `min_support` | 0,05 | igual ao Iris/Pokémon; com ~233 municípios de treino por dobra exige ~12 casos, o mínimo de evidência numa base pequena (a classe Rural tem ~69 casos no treino) |
| `min_confidence` | 0,5 | mesmo da baseline |
| `max_antecedent_length` | 3 | convenção da baseline |
| validação | CV estratificada, 5 dobras, divisão por linha, seed 42 | **o mesmo protocolo** do Mushroom e do German Credit |
| classe positiva | `Rural` | a minoritária (29,5%) |

Limiares escolhidos **a priori** e **não ajustados** olhando o resultado (uma única execução).
Comando: `python -m experimentos.runner --config experimentos/configs/municipios_sc.yaml`.
Execução: `municipios_sc_discretizado_row_m1_20261026T000153527307Z_c2a1c6` (commit `11f6ff3`, **árvore limpa**;
SHA-256 do `municipios_sc.csv` = `041a493249202b5716b62e5991c6d3cc3033ab85a9c407cbef07de7320c25872`; 3 s).
Artefatos em `experimentos/resultados/` (ignorado pelo git); a linha-resumo foi acrescentada a `resumo.csv`.

## 6. Resultados (OOF, 292 predições, uma por linha)

Matriz de confusão (linhas = classe real, colunas = prevista):

|  | prev. `Rural` | prev. `Urbano` |
|---|---:|---:|
| **real `Rural`** (86) | 69 | 17 |
| **real `Urbano`** (206) | 24 | 182 |

| Métrica | CBA M1 | Classe majoritária (`Urbano`) | Referência: copiar `predominance_2010` |
|---|---:|---:|---:|
| Acurácia | **0,8596** | 0,7055 | 0,8904 |
| Acurácia balanceada | 0,8429 | 0,5000 | 0,9189 |
| Macro-F1 | 0,8349 | 0,4137 | 0,8789 |
| MCC | 0,6710 | 0,0000 | 0,7805 |
| Recall `Rural` / `Urbano` | 0,8023 / 0,8835 | 0,0000 / 1,0000 | 0,9884 / 0,8495 |
| Precisão `Rural` / `Urbano` | 0,7419 / 0,9146 | indefinida / 0,7055 | 0,7328 / 0,9943 |

Acurácia por dobra: 0,932 / 0,814 / 0,879 / 0,828 / 0,845 (desvio 0,042). IC 95% (bootstrap por linha, 1000
repetições): acurácia [0,822; 0,897], acurácia balanceada [0,795; 0,888], macro-F1 [0,788; 0,878], MCC [0,580; 0,758].
Regras candidatas (média): 9 626; finais (média): 35,4 (22 a 44 por dobra); antecedente médio: 2,42. Classe padrão
acionada em 21 casos (7,2%).

A coluna "copiar `predominance_2010`" **não é um modelo treinado**: aplica a regra fixa "o município continua com a
classificação de 2010" aos 292 municípios (TP 85, FN 1, FP 31, TN 175). Não tem parâmetro ajustado, portanto não precisa
de validação cruzada. Foi calculada depois da execução, para interpretá-la, e não entrou no protocolo da baseline.

## 7. Auditoria independente

`python -m experimentos.auditoria verify --run experimentos/resultados/municipios_sc_discretizado_row_m1_20261026T000153527307Z_c2a1c6`
→ **APROVADO**: 292 predições OOF (uma por linha); predições reconstruídas só dos cortes e regras salvos idênticas às
OOF (292/292); acurácia, acurácia balanceada, macro-F1, MCC e matriz de confusão recalculados batem com o resumo; em cada
dobra o prefixo do M1 é o primeiro de erro total mínimo (22–44 regras, erro total 3–10 em ~233 casos de treino).

## 8. Leitura honesta do resultado

- **O CBA é muito melhor que a classe majoritária.** +15,4 pontos de acurácia (0,860 contra 0,706), com o IC 95% da
  acurácia [82,2%; 89,7%] **acima** dos 70,5%, e MCC de 0,67 (IC [0,58; 0,76], longe de 0). Diferente do German Credit,
  aqui a acurácia sozinha já separa o modelo da baseline trivial.
- **Mas não é melhor que a regra de uma coluna.** Copiar `predominance_2010` dá 89,0% de acurácia e MCC 0,78, **acima**
  do CBA (86,0% e 0,67). A diferença de ~3 pontos de acurácia está dentro do IC do CBA, então não afirmo que a regra
  trivial seja melhor; afirmo que **o CBA não supera** essa referência. 53% das regras finais (94 de 177 nas 5 dobras)
  usam `predominance_2010` no antecedente, e o CBA coincide com a cópia de 2010 em 91,4% dos municípios. Boa parte do
  que o modelo "aprendeu" é essa persistência histórica.
- **Onde o CBA difere da cópia.** Os erros que importam são os 31 municípios Rural em 2010 e Urbano em 2022 (mudança de
  caráter). O CBA reduz os falsos Rural de 31 para 24, mas ao custo de subir os falsos Urbano de 1 para 17 (os 24 não são
  um subconjunto dos 31: 23 são mudanças não detectadas e 1 é um município que já era Urbano em 2010; portanto o CBA
  acerta 31 − 23 = 8 das 31 mudanças, não 7): o saldo é
  negativo para a classe minoritária (recall Rural 0,80 contra 0,99). Há um sinal modesto sobre as mudanças, não
  suficiente para compensar.
- **Regras sensatas.** As regras de topo são intuitivas e de confiança 100% no treino, como `Population_2022=bin2 =>
  Urbano` (78 de 78) ou `predominance_2010=Urbano AND qnt_pd=bin2 => Urbano` (83 de 83). Não há sinal de vazamento do
  alvo: o resultado não chega a 100% e a dispersão entre dobras (0,81 a 0,93) é compatível com ~58 municípios por dobra.
- **Uma resposta que este teste não dá.** Sem uma execução sem `predominance_2010` não sei quanto o CBA aprende das
  demais colunas. Fica como extensão natural, não feita aqui para manter uma única execução a priori.

## 9. O que este teste cobre e limitações

Cobre dados reais **pequenos** (292 linhas), atributos **mistos** (15 discretizados + 11 categóricos nativos, alguns com
20-33 valores), classe 70/30 e ~9,6 mil regras candidatas por dobra, com um alvo que tem vazamento potencial
tratado explicitamente. Junto do Mushroom (só categórico, trivial), do German Credit (difícil, ruidoso) e do Pokémon
(só contínuo, desbalanceado), fecha os formatos de entrada do pipeline.

Limitações: (i) a divisão é **por linha**, mas municípios vizinhos (mesma `Microrregion`, `Association`) tendem a se
parecer; uma divisão por grupo (A06) seria mais severa e provavelmente daria um valor menor; (ii) o IC por linha com
292 casos é largo; (iii) a autorização de redistribuição é uma mensagem do professor, sem licença formal (seção 2).
`tests/test_municipios_sc.py` é um smoke test: carrega o config real, roda a CV, confere uma predição OOF por linha, a
proporção da classe majoritária (206/292), acurácia abaixo de 100% (um valor perfeito indicaria vazamento) e que a
classe minoritária é predita alguma vez; sem números exatos.
