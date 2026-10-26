# Municípios de Santa Catarina — relatório de consulta para a apresentação

Versão direta do `ADENDO_Municipios_SC.md`, para consultar durante a apresentação. Todos os números vêm da execução
`municipios_sc_discretizado_row_m1_20261026T000153527307Z_c2a1c6` (auditoria `verify`: **APROVADO**) e dos artefatos dela;
nada foi recalculado com outro protocolo.

## Em uma frase

O CBA acerta **86%** dos municípios (a classe majoritária acerta 70,5%), mas **copiar a classificação de 2010 acerta 89%**:
o CBA aprende algo real, só que não supera a persistência temporal.

## 1. Contexto do dataset

- **Origem:** arquivo enviado pelo professor Fábio, da disciplina de mineração de dados, para fins acadêmicos. Não é um
  dataset público (tipo UCI): não há página, DOI nem licença.
- **Permissão de redistribuição: CONFIRMADA em 25/10/2026, por WhatsApp.** O professor Fábio respondeu: "Sim João,
  libero a distribuição da dataset". É uma autorização por mensagem, sem licença formal nem termo de uso. (O `.xlsx`
  original não está no repositório; só o CSV limpo.)
- **Tamanho:** 295 municípios de SC e 42 colunas originais (aba `banco`; as linhas 1-18 são um painel de estatísticas do
  professor, não são dados).
- **Alvo:** `predominance_2022`, Urbano ou Rural. Urbano quando a população em área urbana é pelo menos 50% da população
  total (verificado no dado: vale em 295 de 295 municípios). Split nos 295: **209 Urbano / 86 Rural**. É um alvo binário,
  desbalanceado (~71% / 29%) e com colunas no próprio dado que o entregam de bandeja, o que obriga a decidir o que é
  atributo legítimo e o que é vazamento.

## 2. Pré-processamento

**42 colunas e 295 linhas → 26 atributos + alvo e 292 linhas.**

| O que saiu | Colunas | Por quê (linguagem direta) |
|---|---|---|
| Identificadores (2) | `cities`, `Codigo_IBGE` | nome e código do município não são características; só identificam |
| Entregam o alvo (4) | `Pop_urb`, `Pop_rur`, `dens_urb_22`, `dens_rur` | o alvo é *definido* por `Pop_urb / Population_2022 ≥ 0,5`; as densidades são essa população dividida pela área |
| Quase entregam o alvo (2) | `Domicilios_urb`, `Domicilios_rur` | a mesma divisão urbano/rural, em domicílios |
| Mesma classificação urbano/rural, por área (4) | `area_urb`, `area_rur`, `Urb %`, `Area_Pred` | **não definem o alvo** (`Area_Pred` diz "Rural" para 202 dos 209 municípios Urbanos); saíram por precaução |
| Depende do alvo (1) | `change_2010_2022` | é a comparação entre 2010 e 2022 |
| Faltante demais (2) | `gap_medio_pd` (57% ausentes), `total_instrumentos` (17% ausentes) | imputar 57% é inventar; em `total_instrumentos` nenhuma outra coluna explica o "ausente" e o dicionário nem a define |

**Linhas:** 3 descartadas (`Ouro`, `Pescaria_Brava`, `Balneario_Rincao`), por terem dado faltante em colunas que ficaram.
Todas eram Urbano: sobraram **292 municípios (206 Urbano / 86 Rural)**.

**`predominance_2010` ficou como atributo.** É uma medição histórica, não vem de nenhuma coluna de 2022. Mas é o atributo
que domina o resultado (seções 3 e 4).

**Discretização dos atributos contínuos (15 colunas):** a rotina do projeto, em 3 faixas de igual frequência (tercis),
com os cortes calculados **só nos dados de treino de cada dobra**. Nas 5 dobras todas as colunas ficaram com 3 faixas
(nenhuma colapsou por empates).

**Por que não usei o painel de tercis do professor:**
- O método é o mesmo (conferi: os cortes do painel coincidem com os do projeto em 25 de 26 colunas; a exceção é `IDH_2010`).
- Mas o painel foi calculado **com os 295 municípios**, inclusive os que ficam no teste de cada dobra. Usá-lo faria a
  discretização "enxergar" o teste (vazamento entre dobras). Na rotina do projeto, o teste não participa dos cortes.

**Protocolo (igual ao do Mushroom e do German Credit):** validação cruzada estratificada, 5 dobras, divisão por linha,
seed 42, `min_support` 0,05, `min_confidence` 0,5, antecedente de até 3 itens, escolhidos antes de rodar e **sem ajuste**
(uma única execução).

## 3. Resultado principal

**CBA M1 × classe majoritária × copiar `predominance_2010`** (predições fora da dobra, 292 municípios):

| Métrica | CBA M1 | Classe majoritária (`Urbano`) | Copiar `predominance_2010` |
|---|---:|---:|---:|
| Acurácia | **0,8596** | 0,7055 | **0,8904** |
| Acurácia balanceada | 0,8429 | 0,5000 | **0,9189** |
| MCC | 0,6710 | 0,0000 | **0,7805** |
| Macro-F1 (extra) | 0,8349 | 0,4137 | **0,8789** |

**Intervalos de confiança de 95%** (bootstrap por município, 1000 repetições, só para o CBA):

| Métrica | CBA M1 | IC 95% |
|---|---:|---|
| Acurácia | 0,8596 | [0,822; 0,897] |
| Acurácia balanceada | 0,8429 | [0,795; 0,888] |
| MCC | 0,6710 | [0,580; 0,758] |

A referência "copiar 2010" **não tem IC**: foi calculada só como número pontual, porque é uma regra fixa (não treinada).

**Matriz de confusão do CBA** (linhas = classe real, colunas = prevista):

|  | prev. `Rural` | prev. `Urbano` |
|---|---:|---:|
| **real `Rural`** (86) | 69 | **17** |
| **real `Urbano`** (206) | **24** | 182 |

Por classe: recall `Rural` 0,802 e `Urbano` 0,883; precisão `Rural` 0,742 e `Urbano` 0,915.

**Matriz da referência "copiar 2010"** (para comparar):

|  | prev. `Rural` | prev. `Urbano` |
|---|---:|---:|
| **real `Rural`** (86) | 85 | 1 |
| **real `Urbano`** (206) | 31 | 175 |

**Outros números úteis:** acurácia por dobra 0,932 / 0,814 / 0,879 / 0,828 / 0,845 (desvio 0,042); regras candidatas
~9 626 por dobra, finais 35,4 em média (22 a 44); classe padrão usada em 21 casos (7,2%); tempo total de 3 s.

## 4. Discussão honesta

**1. O CBA bate a majoritária com folga estatística real.** +15,4 pontos de acurácia (0,860 × 0,706). O IC da acurácia
(piso 0,822) fica acima dos 0,7055 da majoritária, e o do MCC ([0,58; 0,76]) está longe de zero. Diferente do German
Credit, aqui a acurácia sozinha já separa o modelo do palpite trivial.

**2. Mas perde para copiar `predominance_2010` em todas as métricas.** Acurácia (0,860 × 0,890), acurácia balanceada
(0,843 × 0,919), MCC (0,671 × 0,781) e macro-F1 (0,835 × 0,879). *Cuidado ao falar:* a diferença de acurácia (~3
pontos) cai **dentro** do IC do CBA, então o correto é dizer que o CBA **não supera** a cópia, não que "foi provado
pior". (A cópia não tem IC.)

**3. Por quê, em termos simples.** Em 12 anos, **a maioria dos municípios não muda de caráter**: 260 dos 292 (89%)
continuam com a mesma classificação de 2010; só 32 mudaram (31 de Rural para Urbano, 1 de Urbano para Rural). Então
"nada mudou" é um palpite muito forte: acerta 89% sem aprender nada. Para ganhar dessa regra, o modelo teria que acertar
**quais** municípios mudam, que é a parte difícil.

**4. O peso de `predominance_2010` nas regras.** **53% das regras finais** o usam (94 de 177, somando as 5 dobras), e o
CBA **concorda com a cópia de 2010 em 91,4%** dos municípios. Boa parte do que o modelo aprendeu é essa persistência.
Nas regras de topo isso aparece de forma direta, por exemplo `predominance_2010=Urbano AND qnt_pd=bin2 => Urbano`
(83 de 83 no treino).

**5. Onde o CBA diverge da cópia: ganha 8 mudanças, perde 16 permanências.** Entre os 116 municípios que eram Rural
em 2010:
- 85 continuaram Rural: o CBA acertou 69 e errou 16 (previu Urbano), todos casos que a cópia acertava;
- 31 viraram Urbano: o CBA acertou **8** e errou 23 (previu Rural, como a cópia).

*Por que 8 e não 7:* os falsos Rural do CBA são 24 (a cópia tem 31), mas os 24 não são um subconjunto dos 31. São 23 mudanças
não detectadas **mais 1** município que já era Urbano em 2010 e que o CBA previu Rural. Então 31 − 23 = 8 mudanças
acertadas; "31 − 24 = 7" mistura os dois grupos.

Entre os 176 que eram Urbano em 2010, o CBA errou 2 (a cópia errou 1). Saldo: +8 − 16 − 1 = −9, que é a diferença entre
os 41 erros do CBA e os 32 da cópia. Há sinal modesto sobre as mudanças (8 de 31 = 26%), mas ele custa mais do que rende.

**6. Não há sinal de vazamento do alvo:** o resultado não passa de 100% e as dobras variam de 0,81 a 0,93, coerente com
~58 municípios por dobra.

## 5. Diagnóstico: os 17 "falsos Urbano"

São os 17 municípios **Rural em 2022 que o CBA classificou como Urbano** (linha `real Rural`, coluna `prev. Urbano`).
Calculado direto das predições OOF salvas e dos nomes da planilha original; nada foi rodado de novo.

**Padrão:**
- **16 dos 17 eram Rural também em 2010**: a cópia de 2010 acertou todos esses 16. O CBA errou onde "nada mudou" estava
  certo. O 17º, `Urupema`, foi Urbano em 2010 e virou Rural (mudança rara, que a cópia também erra).
- **Nenhum veio da classe padrão** (0 de 17): em todos, uma regra específica disparou.
- **São municípios pequenos:** população mediana de 4 245 hab. (a mediana dos Rural é 3 274). Os 3 maiores:
  Gravatal (12 435), Imarui (11 881) e Luiz_Alves (11 684).
- **As regras que erram são, em sua maioria, combinações de 2-3 condições** sobre crescimento, ISS, densidade ou região
  (ex.: `growth_%_2010_2022=bin2 AND issqn=bin0 => Urbano`), com pouca relação óbvia com ser urbano. Isso sugere regras
  que se ajustam a coincidências do treino; é uma leitura, não foi testada.

| Município | Pop. 2022 | Era em 2010 | Regra que decidiu "Urbano" |
|---|---:|---|---|
| Gravatal | 12 435 | Rural | `dens_tot=bin2 AND pop_2010_2022=bin2` |
| Imarui | 11 881 | Rural | `dom_tot=bin2` |
| Luiz_Alves | 11 684 | Rural | `growth_%_2010_2022=bin1 AND issqn=bin2` |
| Alfredo_Wagner | 10 481 | Rural | `itc=bin2` |
| Sao_Joao_do_Sul | 8 668 | Rural | `pop_2010_2022=bin2 AND qnt_pd=bin1` |
| Sao_Joao_do_Oeste | 6 295 | Rural | `Region=Chapeco AND dom_tot=bin1 AND itc=bin1` |
| Botuvera | 5 363 | Rural | `growth_%_2010_2022=bin2 AND growth_2000_2010=bin2` |
| Cordilheira_Alta | 4 781 | Rural | `dens_tot=bin2 AND growth_2000_2010=bin2` |
| Pedras_Grandes | 4 245 | Rural | `Region=Criciuma AND area_tot=bin0` |
| Xavantina | 3 653 | Rural | `Council_PD=Yes AND Region=Chapeco AND area_tot=bin1` |
| Rancho_Queimado | 3 279 | Rural | `IDH_2010=bin2 AND MP=yes AND growth_%_2010_2022=bin2` |
| Atalanta | 3 227 | Rural | `city_birth=bin1 AND dens_tot=bin1 AND growth_%_2010_2022=bin0` |
| Iomere | 2 877 | Rural | `MP=yes AND Microrregion=Joacaba` |
| aguas_Frias | 2 839 | Rural | `growth_%_2010_2022=bin2 AND issqn=bin0` |
| Urupema | 2 656 | **Urbano** | `predominance_2010=Urbano` |
| Tigrinhos | 2 329 | Rural | `growth_%_2010_2022=bin2 AND issqn=bin0` |
| Irati | 2 069 | Rural | `Region=Chapeco AND dens_tot=bin1 AND growth_%_2010_2022=bin0` |

(`binN` = faixa de tercil da coluna: `bin0` baixo, `bin1` médio, `bin2` alto. Os cortes mudam a cada dobra, pois são
ajustados no treino.) Para os 24 "falsos Rural": 23 eram Rural em 2010, ou seja, são municípios que mudaram e que a
cópia também errou; o 24º já era Urbano em 2010 (erro novo do CBA). Por isso o CBA acerta 8 das 31 mudanças (31 − 23),
e não 7 (31 − 24).

## 6. Próximos passos (só citados)

1. **Rodar sem `predominance_2010`**, para ver se o CBA aprende sinal independente da persistência temporal. É o
   experimento que responde à pergunta que este teste deixa em aberto.
2. **Testar a variante M2 do CBA-CB** (poda e cobertura simultâneas). *Atenção:* a M2 **ainda não está implementada** no
   repositório (o runner aceita só `m1` e `cba_qs`), então isso exige implementar antes de testar.

## Limitações a ter em mente

- A divisão é **por linha**, mas municípios vizinhos (mesma microrregião ou associação) se parecem; uma divisão por grupo
  seria mais severa e provavelmente daria um resultado menor.
- Com 292 linhas, os intervalos de confiança são largos.
- Uma execução, limiares a priori, sem ajuste.
- A autorização de redistribuição é uma mensagem do professor, sem licença formal (seção 1).
