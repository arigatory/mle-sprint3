"""Класс FastApiHandler, который обрабатывает запросы API."""

from catboost import CatBoostRegressor


class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запрос и возвращает предсказание."""

    def __init__(self):
        """Инициализация переменных класса."""
        print("Initializing FastApiHandler...")  # Debug log

        # типы параметров запроса для проверки
        self.param_types = {
            "client_id": str,
            "model_params": dict
        }

        model_path = "models/catboost_credit_model.bin"
        self.load_credit_model(model_path=model_path)

        # Получаем список фичей модели после загрузки
        self.feature_names = self.model.feature_names_ if hasattr(
            self.model, 'feature_names_') else None
        if self.feature_names:
            # Debug log
            print(f"Model expects these features: {self.feature_names}")
            self.required_model_params = self.feature_names
        else:
            # Если не удалось получить фичи, используем дефолтный список
            # Debug log
            print("Could not get feature names from model, using default list")
            self.required_model_params = [
                "gender", "Type", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges"
            ]
        # Debug log
        print(f"Required model params: {self.required_model_params}")

    def load_credit_model(self, model_path: str):
        """Загружаем обученную модель предсказания кредитного рейтинга."""
        print(f"Loading model from {model_path}...")  # Debug log
        try:
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
            print("Model loaded successfully")  # Debug log
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def credit_rating_predict(self, model_params: dict) -> float:
        """Предсказываем кредитный рейтинг."""
        print(f"Raw model_params received: {model_params}")  # Debug log

        # Убедимся, что фичи переданы в правильном порядке
        if self.feature_names:
            print("Using model's feature order for prediction")  # Debug log
            input_data = [model_params[feature]
                          for feature in self.feature_names]
        else:
            print("Using default feature order for prediction")  # Debug log
            input_data = list(model_params.values())

        print(f"Prepared input data: {input_data}")  # Debug log
        # Преобразуем в двумерный массив
        return self.model.predict([input_data])[0]

    def check_required_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора."""
        print("\nChecking query params...")  # Debug log
        print(f"Query params received: {query_params}")  # Debug log
        print(f"Expected types: {self.param_types}")  # Debug log

        if "client_id" not in query_params or "model_params" not in query_params:
            # Debug log
            print("Missing required query params: client_id or model_params")
            return False

        if not isinstance(query_params["client_id"], self.param_types["client_id"]):
            # Debug log
            print(
                f"client_id should be {self.param_types['client_id']}, got {type(query_params['client_id'])}")
            return False

        if not isinstance(query_params["model_params"], self.param_types["model_params"]):
            # Debug log
            print(
                f"model_params should be {self.param_types['model_params']}, got {type(query_params['model_params'])}")
            return False

        print("All query params are valid")  # Debug log
        return True

    def check_required_model_params(self, model_params: dict) -> bool:
        """Проверяем, что все обязательные параметры присутствуют."""
        print("\nChecking model params...")  # Debug log
        print(f"Model params received: {model_params}")  # Debug log
        print(f"Required params: {self.required_model_params}")  # Debug log

        missing_params = [
            param for param in self.required_model_params if param not in model_params]
        extra_params = [
            param for param in model_params if param not in self.required_model_params]

        if missing_params:
            print(f"Missing required params: {missing_params}")  # Debug log
        if extra_params:
            print(f"Extra params provided: {extra_params}")  # Debug log

        result = all(
            param in model_params for param in self.required_model_params)
        print(f"Check result: {result}")  # Debug log
        return result

    def validate_params(self, params: dict) -> bool:
        """Проверяем корректность параметров запроса и параметров модели."""
        print("\nValidating all params...")  # Debug log
        query_check = self.check_required_query_params(params)
        print(f"Query params check: {query_check}")  # Debug log

        if not query_check:
            print("Query params validation failed")  # Debug log
            return False

        model_check = self.check_required_model_params(params["model_params"])
        print(f"Model params check: {model_check}")  # Debug log

        if not model_check:
            print("Model params validation failed")  # Debug log
            return False

        print("All params are valid")  # Debug log
        return True

    def handle(self, params):
        """Функция для обработки запросов API."""
        print("\nHandling request...")  # Debug log
        try:
            # Валидируем запрос к API
            if not self.validate_params(params):
                print("Validation failed")  # Debug log
                response = {"Error": "Problem with parameters"}
            else:
                model_params = params["model_params"]
                client_id = params["client_id"]
                print(f"Predicting for client_id: {client_id}")  # Debug log
                print(f"Model params:\n{model_params}")  # Debug log

                # Получаем предсказания модели
                predicted_rating = self.credit_rating_predict(model_params)
                # Debug log
                print(f"Prediction successful: {predicted_rating}")
                response = {
                    "client_id": client_id,
                    "predicted_credit_rating": float(predicted_rating)
                }
        except Exception as e:
            print(f"Error while handling request: {e}")  # Debug log
            return {"Error": str(e)}
        else:
            print("Request handled successfully")  # Debug log
            return response


if __name__ == "__main__":
    print("Starting test...")  # Debug log

    # Создаём тестовый запрос (client_id теперь строка)
    test_params = {
        "client_id": "123",
        "model_params": {
            "gender": 1.0,
            "Type": 0.5501916796819537,
            "SeniorCitizen": 0.0,
            "Partner": 0.0,
            "Dependents": 0.0,
            "PaperlessBilling": 1.0,
            "PaymentMethod": 0.2192247621752094,
            "MonthlyCharges": 50.8,
            "TotalCharges": 288.05,
            "MultipleLines": 0.0,
            "InternetService": 0.3437455629703251,
            "OnlineSecurity": 0.0,
            "OnlineBackup": 0.0,
            "DeviceProtection": 0.0,
            "TechSupport": 1.0,
            "StreamingTV": 0.0,
            "StreamingMovies": 0.0,
            "days": 245.0,
            "services": 2.0
        }
    }

    print(f"Test params:\n{test_params}")  # Debug log

    # создаём обработчик запросов для API
    handler = FastApiHandler()

    # делаем тестовый запрос
    response = handler.handle(test_params)
    print(f"Final response: {response}")  # Debug log
