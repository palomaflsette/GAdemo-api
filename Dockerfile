FROM python:3.12.4

WORKDIR /gademo

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

VOLUME ./src /gademo/src

WORKDIR /gademo/src/api

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
