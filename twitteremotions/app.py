from flask import Flask, request
import flask
from twitteremotions.emotions import TwitterEmotions
from datetime import datetime

app = Flask(__name__)


@app.route("/train")
def train():
    train_path = request.args.get("data", "data/train.csv")
    epochs = request.args.get("epochs", 10)
    emotion.train(train_path, epochs)


@app.route("/predict")
def predict():
    start = datetime.now()
    sentence = request.args.get("sentence")
    sentiment = request.args.get("sentiment")
    response = {}
    response["selected"] = emotion.predict(sentence, sentiment)
    response["time"] = datetime.now() - start

    return flask.jsonify(response)


if __name__ == "main":
    emotion = TwitterEmotions()
