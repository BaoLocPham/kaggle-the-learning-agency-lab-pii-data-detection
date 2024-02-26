import json
from dataset.utils import read_test
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset
from utils import get_logger, seed_everything
from omegaconf import OmegaConf
import hydra
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")


LOGGER = get_logger(
    filename="inference_stage_1",
    logger_name="InferenceLogger")


class CFG:
    root_data_dir = "./data"
    output_dir = "./output"
    output_submission_path = "./"
    debug = True


config = CFG()


def tokenize(example, tokenizer, config):
    text = []
    token_map = []

    idx = 0

    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):

        text.append(t)
        token_map.extend([idx] * len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)

        idx += 1

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=config.return_offsets_mapping,
        truncation=config.truncation,
        max_length=config.max_len,
        stride=config.stride,
        return_overflowing_tokens=config.return_overflowing_tokens
    )

    return {
        **tokenized,
        "token_map": token_map,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer(cfg):
    LOGGER.info(f"{OmegaConf.to_yaml(cfg)}")
    config.__dict__.update(cfg.parameters)

    # If it doesn't exist, create the folder
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.tmp_dir, exist_ok=True)
    test, submission = read_test(
        config.root_data_dir  # type: ignore
    )

    LOGGER.info(f"DEBUG MODE: {config.debug}")

    model_config = json.load(
        open(
            Path(
                config.inference_stage_1.model_path) /
            "config.json"))
    id2label = model_config["id2label"]
    LOGGER.info(f"id2label: {id2label}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.inference_stage_1.model_path)

    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in test],
        "document": [x["document"] for x in test],
        "tokens": [x["tokens"] for x in test],
        "trailing_whitespace": [x["trailing_whitespace"] for x in test],
    })

    ds = ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "config": config.inference_stage_1},
        num_proc=config.num_proc)

    model = AutoModelForTokenClassification.from_pretrained(
        config.inference_stage_1.model_path)
    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=16)
    args = TrainingArguments(
        ".",
        per_device_eval_batch_size=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    #### INFER ####
    predictions = trainer.predict(ds).predictions
    pred_softmax = np.exp(predictions) / np.sum(np.exp(predictions),
                                                axis=2).reshape(predictions.shape[0], predictions.shape[1], 1)

    preds = predictions.argmax(-1)
    preds_without_O = pred_softmax[:, :, :12].argmax(-1)
    O_preds = pred_softmax[:, :, 12]

    threshold = config.inference_stage_1.threshold
    preds_final = np.where(O_preds < threshold, preds_without_O, preds)

    triplets = []
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc in zip(
            preds_final, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):

        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[str(token_pred)]

            if start_idx + end_idx == 0:
                continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(
                    token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map):
                break

            token_id = token_map[start_idx]

            # ignore "O" predictions and whitespace preds
            if label_pred != "O" and token_id != -1:
                triplet = (label_pred, token_id, tokens[token_id])

                if triplet not in triplets:
                    document.append(doc)
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(tokens[token_id])
                    triplets.append(triplet)

    df = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })
    df["row_id"] = list(range(len(df)))
    df[["row_id", "document", "token", "label"]].to_csv(
        config.inference_stage_1.output_path, index=False)


if __name__ == "__main__":
    seed_everything(seed=42)
    infer()
