FROM python:3.12.4

WORKDIR /gademo

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Roda como usuário sem privilégio (antes o processo rodava como root).
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /gademo
USER appuser

EXPOSE 8000

WORKDIR /gademo/src/api

# Sem --reload (modo de desenvolvimento). Produção no Heroku usa o Procfile.
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
