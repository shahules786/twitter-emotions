import tokenizers
import numpy as np
import tensorflow as tf


class Dataprocess:
    def __init(self, text, sentiment, tokenizer):

        self.text = text
        self.tokenizer = tokenizer
        self.sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}
        self.sentiment = sentiment
        self.MAX_LEN = 168

    def preprocess_bert(self):

        input_ids = np.ones((1, self.MAX_LEN), dtype="int32")
        attention_mask = np.zeros((1, self.MAX_LEN), dtype="int32")
        token_type_id = np.zeros((1, self.MAX_LEN), dtype="int32")

        assert type(self.text) == str, "input should be a string"

        self.text = " " + " ".join(self.text.split())
        enc = self.tokenizer.encode(self.text)
        sent_id = self.sentiment_id[self.sentiment]
        input_ids[0, : len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [sent_id] + [2]
        attention_mask[0, : len(enc.ids) + 5] = 1

        return [input_ids, attention_mask, token_type_id]

    def preprocess_output(self, pred_start, pred_end):

        start = np.argmax(pred_start)
        end = np.argmax(pred_end)
        ids = self.tokenizer.encode(" " + " ".join(self.text.split())).ids
        output = self.tokenizer.decode(ids[start - 1 : end])

        return output


class DataGenerator(tf.keras.utils.Sequence):

    """generates data"""

    def __init__(self, df, tokenizer, batch_size=32, shuffle=True, is_test=False, MAX_LEN=168):

        self.MAX_LEN = MAX_LEN
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.df = df
        self.sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}
        self.shuffle = shuffle
        self.is_test = is_test
        self.list_IDs = df.index.values.tolist()
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data

        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        input_ids = np.ones((self.batch_size, self.MAX_LEN), dtype="int32")
        attention_mask = np.zeros((self.batch_size, self.MAX_LEN), dtype="int32")
        token_type_ids = np.zeros((self.batch_size, self.MAX_LEN), dtype="int32")
        start_tokens = np.zeros((self.batch_size, self.MAX_LEN), dtype="int32")
        end_tokens = np.zeros((self.batch_size, self.MAX_LEN), dtype="int32")

        # Generate data
        for k, ID in enumerate(list_IDs_temp):
            # Store sample
            # FIND OVERLAP
            text1 = " " + " ".join(self.df.loc[ID, "text"].split())
            enc = self.tokenizer.encode(text1)

            s_tok = self.sentiment_id[self.df.loc[ID, "sentiment"]]
            input_ids[k, : len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
            attention_mask[k, : len(enc.ids) + 5] = 1

            if not self.is_test:
                text2 = " ".join(self.df.loc[ID, "selected_text"].split())
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
                    start_tokens[k, toks[0] + 1] = 1
                    end_tokens[k, toks[-1] + 1] = 1

                return [input_ids, attention_mask, token_type_ids], [start_tokens, end_tokens]

            else:
                return [input_ids, attention_mask, token_type_ids]
