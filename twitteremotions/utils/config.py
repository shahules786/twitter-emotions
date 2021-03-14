import tokenizers


class Config:
    def __init__(self):

        self.DEVICE = "cuda"
        self.MAX_LEN = 168
        self.PATH = "../input/tf-roberta/"
        self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
            vocab_file=self.PATH + "vocab-roberta-base.json",
            merges_file=self.PATH + "merges-roberta-base.txt",
            lowercase=True,
            add_prefix_space=True,
        )
