# CBA M1 nos municípios de SC — resultados e regras

## 1. Contexto

Dataset dos 295 municípios de Santa Catarina; alvo `predominance_2022` (Urbano/Rural), com 209 Urbano e 86 Rural.
Os resultados abaixo usam **292 municípios** (206 Urbano / 86 Rural), pois 3 foram descartados por dado faltante.
Foram usados 26 atributos: ficaram de fora os identificadores (`cities`, `Codigo_IBGE`), as colunas que definem ou aproximam o alvo
(`Pop_urb`, `Pop_rur`, `Domicilios_urb`, `Domicilios_rur`, `dens_urb_22`, `dens_rur`, `area_urb`, `area_rur`, `Urb %`, `Area_Pred`,
`change_2010_2022`) e as com muitos ausentes (`gap_medio_pd`, `total_instrumentos`); `predominance_2010` **foi mantida** como atributo.

## 2. Resultado

Validação cruzada estratificada, 5 dobras, seed 42; predições fora da dobra (uma por município).

| | Acurácia |
|---|---:|
| **CBA M1** | **86,0%** (0,8596) |
| Classe majoritária (`Urbano`) | 70,5% (0,7055) |

Acurácia balanceada 0,843 e MCC 0,671. Como referência, simplesmente copiar `predominance_2010` como previsão acerta 89,0%.

## 3. Matriz de confusão (CBA M1)

Linhas = classe real, colunas = classe prevista.

|  | prev. `Rural` | prev. `Urbano` |
|---|---:|---:|
| **real `Rural`** (86) | 69 | 17 |
| **real `Urbano`** (206) | 24 | 182 |

## 4. Regras finais por dobra

São **177 regras** (dobra 1: 44, dobra 2: 34, dobra 3: 41, dobra 4: 36, dobra 5: 22). **São 5 classificadores diferentes, um por dobra,
e não um conjunto único de regras**: cada dobra treina com 233 ou 234 municípios e gera o seu, e a mesma regra pode aparecer em várias.

- **Ordem = prioridade do CBA-CB:** maior confiança primeiro; no empate, maior suporte; no empate de novo, a regra gerada antes.
  Para classificar um município vale a **primeira regra da lista cujo antecedente ele satisfaz**; se nenhuma satisfizer, usa-se a
  **classe padrão** da dobra (`Rural` nas 5).
- **Suporte** = (casos do treino que satisfazem o antecedente e têm a classe) ÷ (municípios de treino da dobra).
  **Confiança** = (casos que satisfazem antecedente e classe) ÷ (casos que satisfazem o antecedente), no treino. A fração exata está em "Casos".
- Só estão as regras do classificador final (prefixo escolhido pelo critério de erro mínimo do M1).
- Parâmetros: `min_support` 0,05, `min_confidence` 0,5, antecedente de até 3 itens; discretização de igual frequência em 3 faixas
  (`bin0` baixo, `bin1` médio, `bin2` alto) nos 15 atributos numéricos. **Os cortes de cada dobra estão na última tabela**; sem ela as
  faixas não têm significado. Itens `coluna=valor` são categóricos.

### Dobra 1

44 regras · treino 233 municípios · teste 59 · classe padrão `Rural`

| # | Regra | Suporte | Confiança | Casos (regra / antecedente) |
|---:|---|---:|---:|---|
| 1 | `Population_2022=bin2 => Urbano` | 0,3348 | 1,0000 | 78/78 |
| 2 | `dom_tot=bin2 AND predominance_2010=Urbano => Urbano` | 0,3305 | 1,0000 | 77/77 |
| 3 | `itc=bin2 AND predominance_2010=Urbano => Urbano` | 0,3305 | 1,0000 | 77/77 |
| 4 | `predominance_2010=Urbano AND revenue=bin2 => Urbano` | 0,3305 | 1,0000 | 77/77 |
| 5 | `predominance_2010=Urbano AND transfer=bin2 => Urbano` | 0,3305 | 1,0000 | 77/77 |
| 6 | `predominance_2010=Urbano AND qnt_pd=bin2 => Urbano` | 0,3262 | 1,0000 | 76/76 |
| 7 | `dens_tot=bin2 AND predominance_2010=Urbano => Urbano` | 0,3133 | 1,0000 | 73/73 |
| 8 | `growth_2000_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,3090 | 1,0000 | 72/72 |
| 9 | `iptu=bin2 AND predominance_2010=Urbano => Urbano` | 0,3090 | 1,0000 | 72/72 |
| 10 | `issqn=bin2 AND predominance_2010=Urbano => Urbano` | 0,3004 | 1,0000 | 70/70 |
| 11 | `city_birth=bin1 AND predominance_2010=Urbano => Urbano` | 0,2790 | 1,0000 | 65/65 |
| 12 | `IDH_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,2661 | 1,0000 | 62/62 |
| 13 | `iptu=bin2 AND pop_2010_2022=bin2 => Urbano` | 0,2575 | 1,0000 | 60/60 |
| 14 | `growth_2000_2010=bin2 AND pop_2010_2022=bin2 => Urbano` | 0,2532 | 1,0000 | 59/59 |
| 15 | `itc=bin1 AND predominance_2010=Urbano => Urbano` | 0,2060 | 1,0000 | 48/48 |
| 16 | `growth_%_2010_2022=bin0 AND predominance_2010=Urbano => Urbano` | 0,1674 | 1,0000 | 39/39 |
| 17 | `area_tot=bin0 AND pop_2010_2022=bin2 => Urbano` | 0,1073 | 1,0000 | 25/25 |
| 18 | `growth_%_2010_2022=bin1 AND issqn=bin2 => Urbano` | 0,1073 | 1,0000 | 25/25 |
| 19 | `pop_2010_2022=bin2 AND qnt_pd=bin1 => Urbano` | 0,0944 | 1,0000 | 22/22 |
| 20 | `qnt_pd=bin2 AND year_MP=2023 => Urbano` | 0,0730 | 1,0000 | 17/17 |
| 21 | `Region=Chapeco AND city_birth=bin1 AND itc=bin1 => Urbano` | 0,0687 | 1,0000 | 16/16 |
| 22 | `IDH_2010=bin2 AND Region=Blumenau => Urbano` | 0,0644 | 1,0000 | 15/15 |
| 23 | `dens_tot=bin1 AND growth_%_2010_2022=bin0 AND iptu=bin1 => Urbano` | 0,0644 | 1,0000 | 15/15 |
| 24 | `area_tot=bin0 AND city_birth=bin1 AND growth_%_2010_2022=bin2 => Urbano` | 0,0601 | 1,0000 | 14/14 |
| 25 | `hierarchy=Centro_Sub-Regional_A => Urbano` | 0,0558 | 1,0000 | 13/13 |
| 26 | `Council_PD=non_located AND growth_%_2010_2022=bin0 AND predominance_2010=Rural => Rural` | 0,0515 | 1,0000 | 12/12 |
| 27 | `IDH_2010=bin0 AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,1159 | 0,9643 | 27/28 |
| 28 | `IDH_2010=bin0 AND iptu=bin0 AND predominance_2010=Rural => Rural` | 0,1030 | 0,9600 | 24/25 |
| 29 | `Council_PD=Yes AND dens_tot=bin1 AND dom_tot=bin1 => Urbano` | 0,0944 | 0,9565 | 22/23 |
| 30 | `Council_PD=Yes AND dens_tot=bin1 AND qnt_pd=bin2 => Urbano` | 0,0901 | 0,9545 | 21/22 |
| 31 | `dens_tot=bin0 AND predominance_2010=Rural AND revenue=bin0 => Rural` | 0,1416 | 0,9429 | 33/35 |
| 32 | `IDH_2010=bin2 AND area_tot=bin0 AND growth_%_2010_2022=bin2 => Urbano` | 0,0687 | 0,9412 | 16/17 |
| 33 | `area_tot=bin1 AND predominance_2010=Rural AND transfer=bin0 => Rural` | 0,0644 | 0,9375 | 15/16 |
| 34 | `Region=Chapeco AND city_birth=bin1 AND revenue=bin1 => Urbano` | 0,0601 | 0,9333 | 14/15 |
| 35 | `IDH_2010=bin0 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,1159 | 0,9310 | 27/29 |
| 36 | `city_birth=bin2 AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,1116 | 0,9286 | 26/28 |
| 37 | `city_birth=bin1 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,0558 | 0,9286 | 13/14 |
| 38 | `Region=Blumenau AND hierarchy=Centro_Local AND predominance_2010=Rural => Rural` | 0,0515 | 0,9231 | 12/13 |
| 39 | `dens_tot=bin0 AND predominance_2010=Rural AND qnt_pd=bin2 => Rural` | 0,0515 | 0,9231 | 12/13 |
| 40 | `IDH_2010=bin0 AND dens_tot=bin0 AND itc=bin0 => Rural` | 0,0987 | 0,9200 | 23/25 |
| 41 | `dens_tot=bin0 AND issqn=bin1 AND predominance_2010=Rural => Rural` | 0,0901 | 0,9130 | 21/23 |
| 42 | `area_tot=bin0 AND growth_%_2010_2022=bin2 => Urbano` | 0,1245 | 0,9062 | 29/32 |
| 43 | `issqn=bin1 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,0773 | 0,9000 | 18/20 |
| 44 | `MP=yes AND dens_tot=bin1 AND growth_%_2010_2022=bin0 => Urbano` | 0,0987 | 0,8846 | 23/26 |

### Dobra 2

34 regras · treino 233 municípios · teste 59 · classe padrão `Rural`

| # | Regra | Suporte | Confiança | Casos (regra / antecedente) |
|---:|---|---:|---:|---|
| 1 | `predominance_2010=Urbano AND qnt_pd=bin2 => Urbano` | 0,3562 | 1,0000 | 83/83 |
| 2 | `Population_2022=bin2 => Urbano` | 0,3348 | 1,0000 | 78/78 |
| 3 | `dom_tot=bin2 AND predominance_2010=Urbano => Urbano` | 0,3262 | 1,0000 | 76/76 |
| 4 | `itc=bin2 AND predominance_2010=Urbano => Urbano` | 0,3262 | 1,0000 | 76/76 |
| 5 | `predominance_2010=Urbano AND revenue=bin2 => Urbano` | 0,3262 | 1,0000 | 76/76 |
| 6 | `dens_tot=bin2 AND predominance_2010=Urbano => Urbano` | 0,3219 | 1,0000 | 75/75 |
| 7 | `issqn=bin2 AND predominance_2010=Urbano => Urbano` | 0,3176 | 1,0000 | 74/74 |
| 8 | `growth_2000_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,3090 | 1,0000 | 72/72 |
| 9 | `IDH_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,2790 | 1,0000 | 65/65 |
| 10 | `city_birth=bin1 AND predominance_2010=Urbano => Urbano` | 0,2661 | 1,0000 | 62/62 |
| 11 | `dens_tot=bin2 AND growth_2000_2010=bin2 => Urbano` | 0,2489 | 1,0000 | 58/58 |
| 12 | `growth_%_2010_2022=bin2 AND growth_2000_2010=bin2 => Urbano` | 0,2489 | 1,0000 | 58/58 |
| 13 | `issqn=bin2 AND pop_2010_2022=bin2 => Urbano` | 0,2361 | 1,0000 | 55/55 |
| 14 | `iptu=bin1 AND predominance_2010=Urbano => Urbano` | 0,2146 | 1,0000 | 50/50 |
| 15 | `predominance_2010=Urbano AND revenue=bin1 => Urbano` | 0,2146 | 1,0000 | 50/50 |
| 16 | `issqn=bin1 AND predominance_2010=Urbano => Urbano` | 0,1717 | 1,0000 | 40/40 |
| 17 | `dens_tot=bin0 AND predominance_2010=Rural AND revenue=bin0 => Rural` | 0,1717 | 1,0000 | 40/40 |
| 18 | `city_birth=bin2 AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,1288 | 1,0000 | 30/30 |
| 19 | `IDH_2010=bin0 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,1245 | 1,0000 | 29/29 |
| 20 | `qnt_pd=bin2 AND year_MP=2023 => Urbano` | 0,0815 | 1,0000 | 19/19 |
| 21 | `Council_PD=Yes AND Region=Chapeco AND area_tot=bin1 => Urbano` | 0,0815 | 1,0000 | 19/19 |
| 22 | `Region=Chapeco AND city_birth=bin1 AND dom_tot=bin1 => Urbano` | 0,0687 | 1,0000 | 16/16 |
| 23 | `Region=Chapeco AND city_birth=bin1 AND revenue=bin1 => Urbano` | 0,0644 | 1,0000 | 15/15 |
| 24 | `IDH_2010=bin2 AND Region=Blumenau => Urbano` | 0,0601 | 1,0000 | 14/14 |
| 25 | `city_birth=bin1 AND dens_tot=bin1 AND growth_%_2010_2022=bin0 => Urbano` | 0,0601 | 1,0000 | 14/14 |
| 26 | `Council_PD=non_located AND IDH_2010=bin0 AND predominance_2010=Rural => Rural` | 0,0558 | 1,0000 | 13/13 |
| 27 | `Council_PD=non_located AND growth_%_2010_2022=bin0 AND itc=bin0 => Rural` | 0,0558 | 1,0000 | 13/13 |
| 28 | `IDH_2010=bin2 AND area_tot=bin0 AND growth_%_2010_2022=bin2 => Urbano` | 0,0558 | 1,0000 | 13/13 |
| 29 | `Metropolitan_Region=Contestado AND dom_tot=bin1 => Urbano` | 0,0515 | 1,0000 | 12/12 |
| 30 | `Region=Criciuma AND area_tot=bin0 => Urbano` | 0,0515 | 1,0000 | 12/12 |
| 31 | `Region=Blumenau AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,0515 | 1,0000 | 12/12 |
| 32 | `area_tot=bin2 AND iptu=bin2 => Urbano` | 0,1459 | 0,9714 | 34/35 |
| 33 | `IDH_2010=bin0 AND iptu=bin0 AND predominance_2010=Rural => Rural` | 0,0987 | 0,9583 | 23/24 |
| 34 | `MP=yes AND dens_tot=bin1 AND growth_%_2010_2022=bin0 => Urbano` | 0,0987 | 0,9583 | 23/24 |

### Dobra 3

41 regras · treino 234 municípios · teste 58 · classe padrão `Rural`

| # | Regra | Suporte | Confiança | Casos (regra / antecedente) |
|---:|---|---:|---:|---|
| 1 | `Population_2022=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 2 | `dom_tot=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 3 | `predominance_2010=Urbano AND revenue=bin2 => Urbano` | 0,3248 | 1,0000 | 76/76 |
| 4 | `predominance_2010=Urbano AND transfer=bin2 => Urbano` | 0,3248 | 1,0000 | 76/76 |
| 5 | `predominance_2010=Urbano AND qnt_pd=bin2 => Urbano` | 0,3162 | 1,0000 | 74/74 |
| 6 | `dens_tot=bin2 AND predominance_2010=Urbano => Urbano` | 0,3077 | 1,0000 | 72/72 |
| 7 | `issqn=bin2 AND predominance_2010=Urbano => Urbano` | 0,3077 | 1,0000 | 72/72 |
| 8 | `growth_2000_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,3034 | 1,0000 | 71/71 |
| 9 | `iptu=bin2 AND predominance_2010=Urbano => Urbano` | 0,2949 | 1,0000 | 69/69 |
| 10 | `IDH_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,2778 | 1,0000 | 65/65 |
| 11 | `city_birth=bin1 AND predominance_2010=Urbano => Urbano` | 0,2778 | 1,0000 | 65/65 |
| 12 | `growth_2000_2010=bin2 AND pop_2010_2022=bin2 => Urbano` | 0,2479 | 1,0000 | 58/58 |
| 13 | `iptu=bin1 AND predominance_2010=Urbano => Urbano` | 0,2094 | 1,0000 | 49/49 |
| 14 | `IDH_2010=bin0 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,1197 | 1,0000 | 28/28 |
| 15 | `issqn=bin2 AND qnt_pd=bin1 => Urbano` | 0,1154 | 1,0000 | 27/27 |
| 16 | `area_tot=bin0 AND pop_2010_2022=bin2 => Urbano` | 0,0983 | 1,0000 | 23/23 |
| 17 | `Region=Chapeco AND city_birth=bin1 AND itc=bin1 => Urbano` | 0,0812 | 1,0000 | 19/19 |
| 18 | `area_tot=bin2 AND dens_tot=bin1 => Urbano` | 0,0769 | 1,0000 | 18/18 |
| 19 | `IDH_2010=bin0 AND city_birth=bin2 AND growth_2000_2010=bin0 => Rural` | 0,0726 | 1,0000 | 17/17 |
| 20 | `qnt_pd=bin2 AND year_MP=2023 => Urbano` | 0,0641 | 1,0000 | 15/15 |
| 21 | `city_birth=bin1 AND growth_2000_2010=bin2 AND qnt_pd=bin1 => Urbano` | 0,0641 | 1,0000 | 15/15 |
| 22 | `Population_2022=bin1 AND area_tot=bin0 AND itc=bin1 => Urbano` | 0,0598 | 1,0000 | 14/14 |
| 23 | `Region=Blumenau AND hierarchy=Centro_Local AND predominance_2010=Rural => Rural` | 0,0598 | 1,0000 | 14/14 |
| 24 | `Region=Chapeco AND city_birth=bin1 AND pop_2010_2022=bin1 => Urbano` | 0,0598 | 1,0000 | 14/14 |
| 25 | `area_tot=bin2 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,0598 | 1,0000 | 14/14 |
| 26 | `Metropolitan_Region=Contestado AND dom_tot=bin1 => Urbano` | 0,0556 | 1,0000 | 13/13 |
| 27 | `Region=Criciuma AND area_tot=bin0 => Urbano` | 0,0556 | 1,0000 | 13/13 |
| 28 | `Council_PD=non_located AND IDH_2010=bin0 AND predominance_2010=Rural => Rural` | 0,0556 | 1,0000 | 13/13 |
| 29 | `Council_PD=non_located AND city_birth=bin2 AND growth_%_2010_2022=bin0 => Rural` | 0,0556 | 1,0000 | 13/13 |
| 30 | `growth_%_2010_2022=bin2 AND issqn=bin0 => Urbano` | 0,0513 | 1,0000 | 12/12 |
| 31 | `Council_PD=Yes AND Metropolitan_Region=Contestado AND area_tot=bin1 => Urbano` | 0,0513 | 1,0000 | 12/12 |
| 32 | `dens_tot=bin0 AND predominance_2010=Rural AND revenue=bin0 => Rural` | 0,1410 | 0,9706 | 33/34 |
| 33 | `area_tot=bin2 AND iptu=bin2 => Urbano` | 0,1325 | 0,9688 | 31/32 |
| 34 | `Council_PD=Yes AND dens_tot=bin1 AND qnt_pd=bin2 => Urbano` | 0,1068 | 0,9615 | 25/26 |
| 35 | `Council_PD=Yes AND IDH_2010=bin2 AND issqn=bin0 => Urbano` | 0,0769 | 0,9474 | 18/19 |
| 36 | `dens_tot=bin0 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,1453 | 0,9444 | 34/36 |
| 37 | `IDH_2010=bin0 AND area_tot=bin2 AND predominance_2010=Rural => Rural` | 0,0726 | 0,9444 | 17/18 |
| 38 | `IDH_2010=bin0 AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,1325 | 0,9394 | 31/33 |
| 39 | `Region=Blumenau AND predominance_2010=Rural => Rural` | 0,0641 | 0,9375 | 15/16 |
| 40 | `city_birth=bin1 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,0513 | 0,9231 | 12/13 |
| 41 | `MP=yes AND dens_tot=bin1 AND growth_%_2010_2022=bin0 => Urbano` | 0,0855 | 0,9091 | 20/22 |

### Dobra 4

36 regras · treino 234 municípios · teste 58 · classe padrão `Rural`

| # | Regra | Suporte | Confiança | Casos (regra / antecedente) |
|---:|---|---:|---:|---|
| 1 | `predominance_2010=Urbano AND qnt_pd=bin2 => Urbano` | 0,3376 | 1,0000 | 79/79 |
| 2 | `Population_2022=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 3 | `itc=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 4 | `revenue=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 5 | `transfer=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 6 | `dens_tot=bin2 AND predominance_2010=Urbano => Urbano` | 0,3162 | 1,0000 | 74/74 |
| 7 | `issqn=bin2 AND predominance_2010=Urbano => Urbano` | 0,3077 | 1,0000 | 72/72 |
| 8 | `growth_2000_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,3034 | 1,0000 | 71/71 |
| 9 | `iptu=bin2 AND predominance_2010=Urbano => Urbano` | 0,3034 | 1,0000 | 71/71 |
| 10 | `IDH_2010=bin2 AND predominance_2010=Urbano => Urbano` | 0,2692 | 1,0000 | 63/63 |
| 11 | `growth_%_2010_2022=bin2 AND predominance_2010=Urbano => Urbano` | 0,2692 | 1,0000 | 63/63 |
| 12 | `dens_tot=bin2 AND pop_2010_2022=bin2 => Urbano` | 0,2650 | 1,0000 | 62/62 |
| 13 | `city_birth=bin1 AND predominance_2010=Urbano => Urbano` | 0,2607 | 1,0000 | 61/61 |
| 14 | `growth_%_2010_2022=bin2 AND growth_2000_2010=bin2 AND pop_2010_2022=bin2 => Urbano` | 0,2222 | 1,0000 | 52/52 |
| 15 | `growth_2000_2010=bin1 AND predominance_2010=Urbano => Urbano` | 0,2094 | 1,0000 | 49/49 |
| 16 | `itc=bin1 AND predominance_2010=Urbano => Urbano` | 0,2051 | 1,0000 | 48/48 |
| 17 | `growth_%_2010_2022=bin2 AND issqn=bin2 => Urbano` | 0,1838 | 1,0000 | 43/43 |
| 18 | `issqn=bin1 AND predominance_2010=Urbano => Urbano` | 0,1624 | 1,0000 | 38/38 |
| 19 | `area_tot=bin2 AND iptu=bin2 => Urbano` | 0,1368 | 1,0000 | 32/32 |
| 20 | `IDH_2010=bin0 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,1239 | 1,0000 | 29/29 |
| 21 | `area_tot=bin2 AND dens_tot=bin1 => Urbano` | 0,0855 | 1,0000 | 20/20 |
| 22 | `area_tot=bin2 AND growth_%_2010_2022=bin2 => Urbano` | 0,0855 | 1,0000 | 20/20 |
| 23 | `IDH_2010=bin2 AND Region=Blumenau => Urbano` | 0,0769 | 1,0000 | 18/18 |
| 24 | `Population_2022=bin1 AND Region=Chapeco AND city_birth=bin1 => Urbano` | 0,0769 | 1,0000 | 18/18 |
| 25 | `Region=Chapeco AND city_birth=bin1 AND itc=bin1 => Urbano` | 0,0726 | 1,0000 | 17/17 |
| 26 | `Region=Chapeco AND city_birth=bin1 AND growth_%_2010_2022=bin1 => Urbano` | 0,0641 | 1,0000 | 15/15 |
| 27 | `IDH_2010=bin0 AND area_tot=bin2 AND predominance_2010=Rural => Rural` | 0,0598 | 1,0000 | 14/14 |
| 28 | `area_tot=bin0 AND city_birth=bin1 AND growth_2000_2010=bin2 => Urbano` | 0,0598 | 1,0000 | 14/14 |
| 29 | `qnt_pd=bin2 AND year_MP=2023 => Urbano` | 0,0556 | 1,0000 | 13/13 |
| 30 | `IDH_2010=bin2 AND MP=yes AND growth_%_2010_2022=bin2 => Urbano` | 0,1496 | 0,9722 | 35/36 |
| 31 | `dens_tot=bin0 AND predominance_2010=Rural AND revenue=bin0 => Rural` | 0,1496 | 0,9722 | 35/36 |
| 32 | `dens_tot=bin0 AND growth_%_2010_2022=bin0 AND predominance_2010=Rural => Rural` | 0,1368 | 0,9697 | 32/33 |
| 33 | `city_birth=bin2 AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,1239 | 0,9667 | 29/30 |
| 34 | `MP=yes AND Region=Chapeco AND dom_tot=bin1 => Urbano` | 0,0897 | 0,9545 | 21/22 |
| 35 | `dens_tot=bin0 AND growth_2000_2010=bin0 AND predominance_2010=Rural => Rural` | 0,1453 | 0,9444 | 34/36 |
| 36 | `MP=yes AND Microrregion=Joacaba => Urbano` | 0,0684 | 0,9412 | 16/17 |

### Dobra 5

22 regras · treino 234 municípios · teste 58 · classe padrão `Rural`

| # | Regra | Suporte | Confiança | Casos (regra / antecedente) |
|---:|---|---:|---:|---|
| 1 | `predominance_2010=Urbano => Urbano` | 0,6111 | 1,0000 | 143/143 |
| 2 | `Population_2022=bin2 => Urbano` | 0,3333 | 1,0000 | 78/78 |
| 3 | `IDH_2010=bin0 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,1325 | 1,0000 | 31/31 |
| 4 | `IDH_2010=bin0 AND Population_2022=bin0 AND predominance_2010=Rural => Rural` | 0,1068 | 1,0000 | 25/25 |
| 5 | `Region=Chapeco AND dom_tot=bin1 AND itc=bin1 => Urbano` | 0,1068 | 1,0000 | 25/25 |
| 6 | `area_tot=bin0 AND pop_2010_2022=bin2 => Urbano` | 0,0897 | 1,0000 | 21/21 |
| 7 | `IDH_2010=bin2 AND Region=Blumenau => Urbano` | 0,0769 | 1,0000 | 18/18 |
| 8 | `city_birth=bin1 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,0684 | 1,0000 | 16/16 |
| 9 | `Council_PD=non_located AND IDH_2010=bin0 AND predominance_2010=Rural => Rural` | 0,0598 | 1,0000 | 14/14 |
| 10 | `dens_tot=bin1 AND growth_%_2010_2022=bin0 AND iptu=bin1 => Urbano` | 0,0598 | 1,0000 | 14/14 |
| 11 | `Population_2022=bin0 AND city_birth=bin1 AND predominance_2010=Rural => Rural` | 0,0556 | 1,0000 | 13/13 |
| 12 | `area_tot=bin2 AND iptu=bin0 AND predominance_2010=Rural => Rural` | 0,0556 | 1,0000 | 13/13 |
| 13 | `Metropolitan_Region=Contestado AND itc=bin1 => Urbano` | 0,0513 | 1,0000 | 12/12 |
| 14 | `issqn=bin0 AND itc=bin1 => Urbano` | 0,0513 | 1,0000 | 12/12 |
| 15 | `dens_tot=bin0 AND predominance_2010=Rural AND transfer=bin0 => Rural` | 0,1581 | 0,9737 | 37/38 |
| 16 | `issqn=bin1 AND itc=bin0 AND predominance_2010=Rural => Rural` | 0,0897 | 0,9545 | 21/22 |
| 17 | `dens_tot=bin0 AND issqn=bin1 AND predominance_2010=Rural => Rural` | 0,0855 | 0,9524 | 20/21 |
| 18 | `IDH_2010=bin0 AND issqn=bin0 AND predominance_2010=Rural => Rural` | 0,0812 | 0,9500 | 19/20 |
| 19 | `IDH_2010=bin0 AND dens_tot=bin0 AND predominance_2010=Rural => Rural` | 0,1368 | 0,9412 | 32/34 |
| 20 | `Population_2022=bin0 AND issqn=bin1 AND predominance_2010=Rural => Rural` | 0,0684 | 0,9412 | 16/17 |
| 21 | `dom_tot=bin0 AND issqn=bin1 AND predominance_2010=Rural => Rural` | 0,0684 | 0,9412 | 16/17 |
| 22 | `Region=Chapeco AND dens_tot=bin1 AND growth_%_2010_2022=bin0 => Urbano` | 0,0598 | 0,9333 | 14/15 |

### Cortes da discretização por dobra

`bin0` se valor < corte 1; `bin1` se corte 1 ≤ valor < corte 2; `bin2` se valor ≥ corte 2. Cada célula é `corte 1 ; corte 2`.

| Atributo | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 |
|---|---|---|---|---|---|
| `revenue` | 3.81319e+07 ; 7.7856e+07 | 3.81319e+07 ; 8.36201e+07 | 3.84805e+07 ; 8.05824e+07 | 3.8367e+07 ; 8.09973e+07 | 3.8037e+07 ; 8.45375e+07 |
| `itc` | 2.62792e+06 ; 9.88216e+06 | 2.77756e+06 ; 1.10645e+07 | 2.73965e+06 ; 9.82395e+06 | 2.71834e+06 ; 1.01771e+07 | 2.68184e+06 ; 1.2202e+07 |
| `iptu` | 284335 ; 1.91554e+06 | 336069 ; 1.92933e+06 | 330555 ; 1.89694e+06 | 355016 ; 1.91963e+06 | 320147 ; 1.95886e+06 |
| `issqn` | 480664 ; 2.01798e+06 | 559762 ; 2.74852e+06 | 466009 ; 2.34103e+06 | 545708 ; 2.36599e+06 | 466009 ; 2.42267e+06 |
| `transfer` | 3.25447e+07 ; 5.91566e+07 | 3.2691e+07 ; 5.96629e+07 | 3.30792e+07 ; 6.39189e+07 | 3.29851e+07 ; 6.04318e+07 | 3.29058e+07 ; 6.63664e+07 |
| `city_birth` | 1958 ; 1977 | 1958 ; 1967 | 1958 ; 1984 | 1958 ; 1964.33 | 1958 ; 1964 |
| `growth_2000_2010` | -0.0044 ; 0.121467 | 0.0024 ; 0.108867 | -0.00146667 ; 0.1126 | -0.00146667 ; 0.1112 | 0.00126667 ; 0.114167 |
| `area_tot` | 152.407 ; 314.052 | 165.177 ; 322.682 | 159.872 ; 331.174 | 163.205 ; 315.363 | 159.48 ; 319.501 |
| `dom_tot` | 2124.67 ; 5977.67 | 2327 ; 6170.67 | 2219 ; 6041.33 | 2139.33 ; 6308 | 2069 ; 6814.67 |
| `dens_tot` | 24.1971 ; 50.0686 | 23.7686 ; 50.3933 | 23.6821 ; 48.0003 | 24.6042 ; 50.2343 | 24.7911 ; 51.4883 |
| `Population_2022` | 4665.33 ; 13612.3 | 5222.67 ; 13612.3 | 4809 ; 13560.7 | 4827 ; 14238.3 | 4601.33 ; 15187.3 |
| `pop_2010_2022` | 138 ; 1253 | 125 ; 1241 | 138 ; 1254 | 88 ; 1236 | 84.3333 ; 1234.67 |
| `growth_%_2010_2022` | 0.0231667 ; 0.137867 | 0.0221 ; 0.138333 | 0.0222 ; 0.132933 | 0.0169 ; 0.137733 | 0.0178667 ; 0.127067 |
| `IDH_2010` | 705.333 ; 745.667 | 706 ; 748 | 704.667 ; 743 | 705 ; 743.333 | 704.667 ; 747 |
| `qnt_pd` | 1 ; 2 | 1 ; 2 | 1 ; 2 | 1 ; 2 | 1 ; 2 |

---
*Reprodução:* `python -m experimentos.runner --config experimentos/configs/municipios_sc.yaml --show-rules` (branch `Apresentações`) ·
*dataset:* `datasets/municipios_sc.csv`, SHA-256 `041a493249202b5716b62e5991c6d3cc3033ab85a9c407cbef07de7320c25872` ·
*execução:* `municipios_sc_discretizado_row_m1_20261026T000153527307Z_c2a1c6`.
