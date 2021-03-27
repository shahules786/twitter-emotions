import os
import pandas as pd
import numpy as np
import tokenizers
from utils.model import emotion_model
from utils.dataloader import Dataprocess, DataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging


class TwitterEmotions:
    def __init__(
        self, model_path="data/tf_model.h5", path="data/tf_roberta/", device="cuda", lowercase=True, MAX_LEN=168
    ):

        self.MODEL_PATH = model_path
        self.DEVICE = device
        self.MAX_LEN = MAX_LEN
        self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
            vocab_file=path + "vocab-roberta-base.json",
            merges_file=path + "merges-roberta-base.txt",
            lowercase=lowercase,
            add_prefix_space=True,
        )

    def train(self, train_path="data/train.csv", epochs=10, batch_size=32, max_len=168, test_size=0.25):

        if not os.path.exists(train_path):
            raise FileNotFoundError("Please provide a valid train file path")

        df = pd.read_csv(train_path)
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        train_datagen = DataGenerator(train, tokenizer=self.TOKENIZER, batch_size=batch_size, is_test=False)
        test_datagen = DataGenerator(test, tokenizer=self.TOKENIZER, batch_size=batch_size, is_test=False)

        model = emotion_model()

        sv = tf.keras.callbacks.ModelCheckpoint(
            "roberta_model.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        model.fit(train_datagen, epochs=epochs, validation_data=test_datagen, callbacks=[sv], verbose=1)

        logging.info("Model trained succesfully")

    def predict(self, text, sentimemt):

        data = Dataprocess(text, sentimemt, self.TOKENIZER)
        input_ids, attention_mask, token_typeids = data.preprocess_bert()
        model = emotion_model()
        model.load_weights(os.path.join(self.MODEL_PATH, "roberta_model.h5"))
        start, end = model.predict([input_ids, attention_mask, token_typeids])
        output = data.preprocess_output(start, end)
        return output
