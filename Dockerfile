FROM python:3.8.19

WORKDIR /gademo

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

VOLUME ./src/app /gademo/src/app

WORKDIR /gademo/src/app/interface

ENTRYPOINT ["uvicorn", "src.app.interface.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
