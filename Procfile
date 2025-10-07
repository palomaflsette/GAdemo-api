web: gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -t 30 -b 0.0.0.0:$PORT src.api.main:app
worker: celery -A src.api.tasks.celery_app worker --loglevel=info -c 1