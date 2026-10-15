# Ambiente isolado com o pyARC real (e o pyfim, minerador em C que ele exige), usado por
# experimentos/comparar_pyarc.py. O pacote `pyarc` declara dependência do `sklearn` (pacote
# obsoleto que falha ao instalar), por isso é instalado com --no-deps.
FROM python:3.11
RUN pip install --no-cache-dir "numpy<2" "pandas<2.1" scikit-learn pyfim matplotlib \
 && pip install --no-cache-dir --no-deps pyarc
WORKDIR /work
