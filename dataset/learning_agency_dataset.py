import torch
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np


def tokenize(example, tokenizer, label2id, max_length, config: None):
    text = []

    # these are at the character level
    labels = []
    targets = []

    for t, l, ws in zip(
            example["tokens"], example["provided_labels"], example["trailing_whitespace"]):

        text.append(t)
        labels.extend([l] * len(t))

        if l in config.target:
            targets.append(1)
        else:
            targets.append(0)
        # if there is trailing whitespace
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length)

    target_num = sum(targets)
    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:

        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num > 0 else 0
    }
