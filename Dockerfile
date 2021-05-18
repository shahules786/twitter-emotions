FROM python:3.8-buster

COPY . /emotions

WORKDIR /emotions

RUN pip3 install -r requirements.txt

ADD https://github.com/shahules786/twitter-emotions/releases/download/v1.0.0/pytorch_model.bin ./data/roberta/

ADD https://github.com/shahules786/twitter-emotions/releases/download/v1.0.0/emotion_torch.pth ./data/

ENV PYTHONPATH="$PYTHONPATH:/emotions"
