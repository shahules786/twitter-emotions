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
    sentiment = request.args.get("sentiment", "neutral")
    response = {}
    response["selected"] = emotion.predict(sentence, sentiment)
    response["time"] = str(datetime.now() - start)

    return flask.jsonify(response)


if __name__ == "__main__":
    emotion = TwitterEmotions()
    app.run(host="0.0.0.0", port="9999", debug=True)
