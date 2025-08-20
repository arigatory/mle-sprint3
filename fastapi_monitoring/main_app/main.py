from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np

# создание экземпляра FastAPI приложения
app = FastAPI()

# инициализируем и запускаем экпортёр метрик
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# предсказания
@app.get("/predict")
def predict(x: int, y: int):
    #print(x)
    np.random.seed(int(abs(x)))
    prediction = x+y + np.random.normal(0,1)
    return {'prediction': prediction}
