"""Приложение Fast API для модели предсказания кредитного рейтинга."""

from fastapi import FastAPI, Body
from .fast_api_handler import FastApiHandler

# создаём приложение FastAPI
app = FastAPI()

# создаём обработчик запросов для API
app.handler = FastApiHandler()


@app.post("/api/credit/")
def is_credit_approved(client_id: str, model_params: dict):
    """Функция определяет, выдать кредит или нет, на основании кредитного рейтинга клиента.

    Args:
        client_id (str): Идентификатор клиента.
        model_params (dict): Произвольный словарь с параметрами для модели.

    Returns:
        dict: Предсказание, выдаётся ли кредит.
    """

    all_params = {
        "client_id": client_id,
        "model_params": model_params
    }
    user_prediction = app.handler.handle(all_params)  # обращаемся к модели
    score = user_prediction["predicted_credit_rating"]  # получаем score
    if score >= 600:  # сравниваем с порогом
        approved = 1  # положительное решение, если выше
    else:
        approved = 0  # отрицательное решение, если ниже
    return {"client_id": client_id, "approved": approved}  # ответ
