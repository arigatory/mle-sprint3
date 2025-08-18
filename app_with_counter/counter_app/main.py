import random
from fastapi import FastAPI


# создание экземпляра FastAPI приложения
app = FastAPI()

# standalone режим - один инстанс приложения
# глобальная переменная app.counter имеет одно и то же значение 
# для всего приложения
app.type = "00"
app.counter = 0


# endpoint для счетчика запросов
@app.get("/count")
def count():
    app.counter += 1

# получить посчитанное число запросов
@app.get("/get_counts")
def count():
    return {'counts': app.counter}
