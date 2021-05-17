FROM python:3.8-buster

COPY . /emotions

WORKDIR /emotions

#RUN pip3 install -r requirements.txt

ADD https://huggingface.co/roberta-base/resolve/main/merges.txt ./data/roberta/
ADD https://huggingface.co/roberta-base/resolve/main/vocab.json ./data/roberta/



