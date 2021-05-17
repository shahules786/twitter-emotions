import os
import pandas as pd
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from twitteremotions.utils.model import EmotionModel
from twitteremotions.utils.dataloader import Dataprocess, EmotionData
from twitteremotions.utils.engine import train_fn, eval_fn

import torch
import logging


class TwitterEmotions:
    def __init__(self, model_path="data/roberta/", path="data/", device="cuda", lowercase=True, MAX_LEN=168):
        self.MODEL_PATH = model_path
        self.DEVICE = device
        self.MAX_LEN = MAX_LEN
        self.TOKENIZER = ByteLevelBPETokenizer(
            vocab_file=path + "vocab.json",
            merges_file=path + "merges.txt",
            lowercase=lowercase,
            add_prefix_space=True,
        )

        self.model = EmotionModel()
        DEVICE = torch.device("cpu")
        self.model.load_state_dict(
            torch.load(os.path.join(self.MODEL_PATH, "emotion_torch.pth"), map_location=DEVICE),
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

        # num_train_steps = int(len(train) / batch_size * epochs)
        optimizer = AdamW(model.parameters(), lr=3e-3)

        best_loss = np.inf
        for epoch in range(epochs):
            train_loss = train_fn(train_dataloader, model, optimizer, DEVICE)
            valid_loss = eval_fn(test_dataloader, model, DEVICE)

            if valid_loss < best_loss:
                torch.save(model.state_dict(), "roberta_model.pth")

            print(f"Epoch:{epoch}  train loss -- > {train_loss : .3f}  valid loss --> {valid_loss : .3f}")

        logging.info("Model trained succesfully")

    def predict(self, text, sentimemt):

        data = Dataprocess(text, sentimemt, self.TOKENIZER)
        input_ids, attention_mask = data.preprocess_bert()
        start, end = self.model(input_ids, attention_mask)
        output = data.preprocess_output(start, end)
        return output
