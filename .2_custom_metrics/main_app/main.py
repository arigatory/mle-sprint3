from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from prometheus_client import Histogram
from prometheus_client import Counter
# ваш код здесь — необходимый импорт

# создание экземпляра FastAPI-приложения
app = FastAPI()

# инициализируем и запускаем экпортёр метрик
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

main_app_predictions = Histogram(
    # имя метрики
    "main_app_predictions",
    # описание метрики
    "Histogram of predictions",
    # указываем корзины для гистограммы
    buckets=(1, 2, 4, 5, 10)
)


# ваш код здесь — объект для сбора метрики
# объект для сбора метрики-счётчика
positive_predictions_counter = Counter(
    # имя метрики
    "positive_predictions_total",
    # описание метрики
    "Total number of positive predictions"
)


@app.get("/predict")
def predict(x: int, y: int):
    np.random.seed(x)
    prediction = x+y + np.random.normal(0,1)
    main_app_predictions.observe(prediction)
    
    # ваш код здесь — увеличение метрики счётчика
    # увеличение метрики счётчика, если предсказание больше нуля
    if prediction > 0:
        positive_predictions_counter.inc()
    
    return {'prediction': prediction}
