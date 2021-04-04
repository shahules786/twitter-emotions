import tokenizers
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataprocess:
    def __init(self, text, sentiment, tokenizer):

        self.text = text
        self.tokenizer = tokenizer
        self.sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}
        self.sentiment = sentiment
        self.MAX_LEN = 168

    def preprocess_bert(self):

        input_ids = np.ones((self.MAX_LEN), dtype="int32")
        attention_mask = np.zeros((self.MAX_LEN), dtype="int32")

        assert type(self.text) == str, "input should be a string"

        self.text = " " + " ".join(self.text.split())
        enc = self.tokenizer.encode(self.text)
        sent_id = self.sentiment_id[self.sentiment]
        input_ids[: len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [sent_id] + [2]
        attention_mask[: len(enc.ids) + 5] = 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def preprocess_output(self, pred_start, pred_end):

        start = np.argmax(pred_start)
        end = np.argmax(pred_end)
        ids = self.tokenizer.encode(" " + " ".join(self.text.split())).ids
        output = self.tokenizer.decode(ids[start - 1 : end])

        return output


class EmotionData(Dataset):

    """generates data"""

    def __init__(self, df, tokenizer, is_test=False, MAX_LEN=168):

        self.MAX_LEN = MAX_LEN
        self.df = df
        self.sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}
        self.is_test = is_test
        self.list_IDs = df.index.values.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.df)

    def __getitem__(self, index):

        input_ids = np.ones((self.MAX_LEN), dtype="int32")
        attention_mask = np.zeros((self.MAX_LEN), dtype="int32")

        start_tokens = np.zeros((self.MAX_LEN), dtype="int32")
        end_tokens = np.zeros((self.MAX_LEN), dtype="int32")

        text1 = " " + " ".join(self.df.loc[index, "text"].split())

        enc = self.tokenizer.encode(text1)

        s_tok = self.sentiment_id[self.df.loc[index, "sentiment"]]
        input_ids[: len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask[: len(enc.ids) + 5] = 1

        if not self.is_test:

            text2 = " ".join(self.df.loc[index, "selected_text"].split())
            idx = text1.find(text2)
            chars = np.zeros((len(text1)))
            chars[idx : idx + len(text2)] = 1
            if text1[idx - 1] == " ":
                chars[idx - 1] = 1
            offsets = enc.offsets
            toks = []
            for i, (a, b) in enumerate(offsets):
                sm = np.sum(chars[a:b])
                if sm > 0:
                    toks.append(i)
            if len(toks) > 0:
                start_tokens[toks[0] + 1] = 1
                end_tokens[toks[-1] + 1] = 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "start_tokens": torch.tensor(start_tokens, dtype=torch.long),
            "end_tokens": torch.tensor(end_tokens, dtype=torch.long),
        }
