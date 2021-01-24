import random
import json
from lxml import html
import requests
from newsapi import NewsApiClient
import wikipedia
import pyowm
import webbrowser
from google_speech import Speech
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jarvis"
print("Let's chat! (type 'quit' to exit)")


def actions(ques,q):
    if 'wiki' in ques['tag']:
        z=q
        print(wikipedia.summary(str(z.split(' ',1)[1]),sentences=2))

    if'news' in ques:
        print('TOP NEWS : ')
        newsapi = NewsApiClient(api_key='439908f4580845f696e6a19fa868cfd5')
        top_headlines = newsapi.get_top_headlines(language='en')

        print(top_headlines)

    if 'google' in ques['tag']:
        new=2
        tabUrl="http://google.com/?#q="
        webbrowser.open(tabUrl+q,new=new)

    if 'weather' in ques['tag']:
        owm=pyowm.OWM('59927012dd656297a866fe1dc096294b')
        mgr = owm.weather_manager()
        observation=mgr.one_call(lat=35.7595,lon=-5.8340).forecast_daily[0].temperature('celsius').get('feels_like_morn', None)
        print(f"{bot_name}: {observation} C")
lang = "en"

while True:
    # sentence = "What is MBD ?"
    sentence = input("You: ")
    q = sentence
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                val = random.choice(intent['responses'])
                print(f"{bot_name}: {val}")
                #speech = Speech(val, lang)
                #sox_effects = ("speed", "1.0")
                #speech.play(sox_effects)
                if 'context_set' in intent:
                    actions(intent,q)

    else:
        print(f"{bot_name}: Right now i'm in developing stage so I can't respond on every healthcare problem as soon i'm developed, I can do everything...")