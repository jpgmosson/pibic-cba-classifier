"""CBA: Classification Based on Associations.

Este módulo conterá a implementação própria das duas etapas centrais do
algoritmo CBA:

- CBA-RG (Rule Generator): geração das regras de classificação candidatas a
  partir dos itemsets frequentes minerados (via Apriori ou FP-Growth) e das
  métricas de suporte e confiança.
- CBA-CB (Classifier Builder): construção do classificador final a partir das
  regras geradas, nas variantes M1 (ordenação e cobertura de dados sequencial)
  e M2 (mais eficiente, com poda de regras).
"""
