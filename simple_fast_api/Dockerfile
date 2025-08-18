FROM python:3.11-slim

LABEL author=${AUTHOR}
# ${AUTHOR} - это переменная среды

COPY . ./simple_fast_api
WORKDIR simple_fast_api 
RUN pip3 install -r requirements.txt
VOLUME /speeches

EXPOSE ${APP_PORT}
# указываем переменную среды

CMD uvicorn app.main:app --host  0.0.0.0 --port ${APP_PORT}
# изменили команду на простую строку