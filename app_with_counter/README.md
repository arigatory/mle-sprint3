### 2 Микросервиса:

1) Микросервис для общения с коровой: 
Корова выбирает приветствия из файла ./speeches/greetings.txt 
и записывает, и каждый раз как к нему обратились отправляет запрос на второй микросервис

2) Микросервис-счетчик:
Считает, сколько раз к нему обратились.
Возвращает число обращений по запросу




###  Для запуска локально:

```bash
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements_counter.txt
pip3 install -r requirements_main.txt
uvicorn main_app.main:app --host 0.0.0.0 --port 1702
```

в новом терминале перейдите в директорию, где лежит этот README.MD
```bash
source venv/bin/activate
uvicorn counter_app.main:app --host 0.0.0.0 --port 4649
```

### Для клиента:

```bash
#общение с коровой
curl http://127.0.0.1:1702/cowsay?input_=Muuu!
#сколько раз обратились к корове?
curl http://127.0.0.1:4649/get_counts
```
cd

### Запуск в Docker
-  build

```bash
docker image build . --file Dockerfile_counter --tag app_with_counter:counter
docker image build . --file Dockerfile_main --tag app_with_counter:main
```

- run

```bash
docker container run --name main_app -d --network host  --env-file .env app_with_counter:main
docker container run --publish 4649:1602 --name counter -d --env-file .env app_with_counter:counter
```


