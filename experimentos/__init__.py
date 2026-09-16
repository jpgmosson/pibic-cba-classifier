"""Ambiente de experimentos: roda o pipeline completo do CBA (discretização
+ CBA-RG + CBA-CB) sobre um dataset a partir de um arquivo de configuração,
com validação cruzada estratificada e registro de metadados de
reprodutibilidade (semente, partições, tempo, versões de dependências e
quantidade de regras).

Esta pasta é deliberadamente separada de ``tests/`` (que testa as funções
de ``fase2_implementacao/`` isoladamente, com dados artificiais pequenos) e
de ``fase2_implementacao/`` (implementação pura do CBA, sem lógica de
execução de experimento ou de dataset específico).
"""
