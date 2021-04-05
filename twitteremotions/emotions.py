import os
import pandas as pd
import numpy as np
from tokenizers import AdamW, get_linear_schedule_with_warmup, ByteLevelBPETokenizer
from utils.model import EmotionModel
from utils.dataloader import Dataprocess, EmotionData
from torch.utils.data import DataLoader
from utils.engine import train_fn, eval_fn
from sklearn.model_selection import train_test_split
import torch
import logging


class TwitterEmotions:
    def __init__(self, model_path="data/roberta/", path="data/roberta/", device="cuda", lowercase=True, MAX_LEN=168):

        self.MODEL_PATH = model_path
        self.DEVICE = device
        self.MAX_LEN = MAX_LEN
        self.TOKENIZER = ByteLevelBPETokenizer(
            vocab_file=path + "vocab.json",
            merges_file=path + "merges.txt",
            lowercase=lowercase,
            add_prefix_space=True,
        )

    def train(self, train_path="data/train.csv", epochs=10, batch_size=32, max_len=168, test_size=0.25):

        if not os.path.exists(train_path):
            raise FileNotFoundError("Please provide a valid train file path")

        df = pd.read_csv(train_path).fillna("")
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        train_dataloader = DataLoader(EmotionData(train), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(EmotionData(test, is_test=True), batch_size=batch_size, shuffle=True)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = EmotionModel()
        model.to(DEVICE)

        num_train_steps = int(len(train) / batch_size * epochs)
        optimizer = AdamW(model.parameters(), lr=3e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        best_loss = np.inf
        for epoch in range(epochs):
            train_loss = train_fn(train_dataloader, model, optimizer, DEVICE, scheduler)
            valid_loss = eval_fn(test_dataloader, model, DEVICE)

            if valid_loss < best_loss:
                torch.save(model.state_dict(), "model.pt")

            print(f"Epoch:{epoch}  train loss -- > {train_loss : .3f}  valid loss --> {valid_loss : .3f}")

        logging.info("Model trained succesfully")

    def predict(self, text, sentimemt):

        data = Dataprocess(text, sentimemt, self.TOKENIZER)
        input_ids, attention_mask, token_typeids = data.preprocess_bert()
        model = EmotionModel()
        print(model.summary())
        model.load_weights(os.path.join(self.MODEL_PATH, "tf_model.h5"))
        start, end = model.predict([input_ids, attention_mask, token_typeids])
        output = data.preprocess_output(start, end)
        return output
