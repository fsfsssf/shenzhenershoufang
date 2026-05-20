FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD exec gunicorn --bind :80 --workers 1 --threads 8 --timeout 300 app:app
