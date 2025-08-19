import random
import requests

from cowsay import cowsay
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse


def get_greetings():
    with open('speeches/greetings.txt') as fd:
        greetings = fd.readlines()
    return greetings

# создание экземпляра FastAPI приложения
app = FastAPI()

# обработка запросов к корове
@app.get("/cowsay", response_class=PlainTextResponse)
def cow_answer(input_: str):
    greetings_variants = get_greetings()
    
    greetings_choosen = random.choice(greetings_variants)
    answer = cowsay(greetings_choosen+' '+input_) + '\n'
    url = 'http://counter:1602/count'
    ret = requests.get(url)
    print(ret.json())
    return answer
