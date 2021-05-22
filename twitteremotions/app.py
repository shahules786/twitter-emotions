from flask import Flask, request, render_template
import flask
from twitteremotions.emotions import TwitterEmotions
from datetime import datetime

app = Flask(__name__)


@app.route("/train")
def train():
    train_path = request.args.get("data", "data/train.csv")
    epochs = request.args.get("epochs", 10)
    emotion.train(train_path, epochs)


@app.route("/", methods=["POST", "GET"])
def predict():

    if request.method == "POST":
        start = datetime.now()
        sentence = request.form.get("tweet")
        sentiment = request.form.get("Sentiment", "neutral")
        response = {}
        response["tweet"] = sentence
        response["sentiment"] = sentiment
        response["selected"] = emotion.predict(sentence, sentiment)
        response["time"] = str(datetime.now() - start)
        return render_template("index.html", prediction=response["selected"])

    return render_template("index.html")


if __name__ == "__main__":
    emotion = TwitterEmotions()
    app.run(host="0.0.0.0", port="9999", debug=True)
