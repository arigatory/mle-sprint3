Микросервис для общения с коровой!
Корова выбирает приветствия из файла ./speeches/greetings.txt 


Для запуска:
- python3 -m venv ./venv
- source venv/bin/activate
- pip3 install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 1702

Для проверки:
- curl http://127.0.0.1:1702/cowsay?input_=Muuu!