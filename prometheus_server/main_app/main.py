from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from prometheus_client import Histogram
from prometheus_client import Counter

# создание экземпляра FastAPI приложения
app = FastAPI()



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

main_app_counter_pos = Counter("main_app_counter_pos", "Count of positive predictions")

# предсказания
@app.get("/predict")
def predict(x: int, y: int):
    np.random.seed(int(abs(x)))
    prediction = x+y + np.random.normal(0,1)
    main_app_predictions.observe(prediction)
    if prediction > 0:
        main_app_counter_pos.inc()
    return {'prediction': prediction}
