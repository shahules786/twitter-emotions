import tokenizers
import numpy as np
from config import Config

conf = Config()
MAX_LEN = conf.MAX_LEN
PATH = conf.PATH


class Dataprocess:
    def __init(self, text, sentiment):

        self.text = text
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=PATH + "vocab-roberta-base.json",
            merges_file=PATH + "merges-roberta-base.txt",
            lowercase=True,
            add_prefix_space=True,
        )
        self.sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}
        self.sentiment = sentiment

    def preprocess_bert(self):

        input_ids = np.ones((1, MAX_LEN), dtype="int32")
        attention_mask = np.zeros((1, MAX_LEN), dtype="int32")
        token_type_id = np.zeros((1, MAX_LEN), dtype="int32")

        assert type(self.text) == str, "input should be a string"

        self.text = " " + " ".join(self.text.split())
        enc = self.tokenizer.encode(self.text)
        sent_id = self.sentiment_id[self.sentiment]
        input_ids[0, : len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [sent_id] + [2]
        attention_mask[0, : len(enc.ids) + 5] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_id": token_type_id}

    def preprocess_output(self, pred_start, pred_end):

        start = np.argmax(pred_start)
        end = np.argmax(pred_end)
        ids = self.tokenizer.encode(" " + " ".join(self.text.split())).ids
        output = self.tokenizer.decode(ids[start - 1 : end])

        return output
