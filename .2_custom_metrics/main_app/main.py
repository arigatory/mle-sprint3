from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from prometheus_client import Histogram

# создание экземпляра FastAPI приложения
app = FastAPI()

np.random.seed(1702)

# инициализируем и запускаем экпортёр метрик
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

main_app_predictions = Histogram(
    # имя метрики
    "main_app_predictions",
    #описание метрики
    "Histogram of predictions",
    #указаываем корзины для гистограммы
    buckets=(1, 2, 4, 5, 10)
)

# предсказания
@app.get("/predict")
def predict(x: int, y: int):
    np.random.seed(int(abs(x)))
    prediction = x+y + np.random.normal(0,1)
    main_app_predictions.observe(prediction)
    return {'prediction': prediction}
