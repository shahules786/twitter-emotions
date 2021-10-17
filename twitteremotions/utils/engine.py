import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device):
    model.train()
    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        start, end = model(data["input_ids"], data["attention_mask"])
        loss = loss_fn(
            start, end, torch.argmax(data["start_tokens"], axis=1), torch.argmax(data["end_tokens"], axis=1)
        )
        loss.backward()
        optimizer.step()
        final_loss += loss.item()

    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        start, end = model(data["input_ids"], data["attention_mask"])
        loss = loss_fn(
            start, end, torch.argmax(data["start_tokens"], axis=1), torch.argmax(data["end_tokens"], axis=1)
        )
        final_loss += loss.item()

    return final_loss / len(data_loader)
