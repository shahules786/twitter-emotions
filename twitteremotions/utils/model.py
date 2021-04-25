import torch.nn as nn
import torch
from transformers import RobertaModel, RobertaConfig


class EmotionModel(nn.Module):
    def __init__(self, PATH="data/tf_roberta/"):
        super(EmotionModel, self).__init__()
        config = RobertaConfig.from_pretrained(PATH + "config.json", return_dict=False)
        self.bert_model = RobertaModel.from_pretrained(PATH + "pytorch_model.bin", config=config)
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 1)
        self.linear2 = nn.Linear(768 + 1, 1)

    def forward(self, input_ids, attention_mask):

        out, _ = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out)
        x2b = self.linear1(x)
        x2 = torch.flatten(x2b, 1)

        x1 = torch.cat((out, x2b), 2)
        x1 = self.linear2(x1)
        x1 = torch.flatten(x1, 1)

        return x1, x2
