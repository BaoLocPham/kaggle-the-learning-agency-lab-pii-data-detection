import json
import re
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
        padding=config.padding,
        max_length=config.max_len,
        stride=config.stride,
        return_overflowing_tokens=config.return_overflowing_tokens
    )

    return {
        **tokenized,
        "token_map": token_map,
    }


def backwards_map_preds(i, sub_predictions, max_len, stride):
    if max_len != 1:  # nothing to map backwards if sequence is too short to be split in the first place
        if i == 0:
            # First sequence needs no SEP token (used to end a sequence)
            sub_predictions = sub_predictions[:, :-1, :]
        elif i == max_len - 1:
            # End sequence needs to CLS token + Stride tokens
            # CLS tokens + stride tokens
            sub_predictions = sub_predictions[:, 1 + stride:, :]
        else:
            # Middle sequence needs to CLS token + Stride tokens + SEP token
            sub_predictions = sub_predictions[:, 1 + stride:-1, :]
    return sub_predictions


def backwards_map_(i, row_attribute, max_len, stride):
    # Same logics as for backwards_map_preds - except lists instead of 3darray
    if max_len != 1:
        if i == 0:
            row_attribute = row_attribute[:-1]
        elif i == max_len - 1:
            row_attribute = row_attribute[1 + stride:]
        else:
            row_attribute = row_attribute[1 + stride:-1]
    return row_attribute


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

    if config.inference_stage_1.stride != 0:
        preds = []
        ds_dict = {
            "document": [],
            "token_map": [],
            "offset_mapping": [],
            "tokens": []
        }

        for row in ds:
            # keys that need to be re-assembled
            row_preds = []
            row_offset = []

            for i, y in enumerate(row["offset_mapping"]):
                # create new datasset for each of of the splits per document
                x = Dataset.from_dict({
                    "token_type_ids": [row["token_type_ids"][i]],
                    "input_ids": [row["input_ids"][i]],
                    "attention_mask": [row["attention_mask"][i]],
                    "offset_mapping": [row["offset_mapping"][i]]
                })
                # predict for that split
                pred = trainer.predict(x).predictions
                # removing the stride and additional CLS & SEP that are created
                row_preds.append(
                    backwards_map_preds(
                        i, pred, len(
                            row["offset_mapping"]), config.inference_stage_1.stride))
                row_offset += backwards_map_(i,
                                             y,
                                             len(row["offset_mapping"]),
                                             config.inference_stage_1.stride)

            # Finalize row
            ds_dict["document"].append(row["document"])
            ds_dict["tokens"].append(row["tokens"])
            ds_dict["token_map"].append(row["token_map"])
            ds_dict["offset_mapping"].append(row_offset)

            # Finalize prediction collection by concattenating
            p_concat = np.concatenate(row_preds, axis=1)
            preds.append(p_concat)
        preds_final = []
        for predictions in preds:
            predictions_softmax = np.exp(predictions) / np.sum(
                np.exp(predictions), axis=2).reshape(
                predictions.shape[0], predictions.shape[1], 1)
            predictions = predictions.argmax(-1)
            predictions_without_O = predictions_softmax[:, :, :12].argmax(-1)
            O_predictions = predictions_softmax[:, :, 12]

            threshold = 0.9
            preds_final.append(
                np.where(
                    O_predictions < threshold,
                    predictions_without_O,
                    predictions))
        ds = Dataset.from_dict(ds_dict)
        pairs = []
        document, token, label, token_str = [], [], [], []
        processed = []
        for p, token_map, offsets, tokens, doc in zip(
                preds_final, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):
            for token_pred, (start_idx, end_idx) in zip(p[0], offsets):
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
                if label_pred in (
                    "O",
                    "B-EMAIL",
                    "B-PHONE_NUM",
                        "I-PHONE_NUM") and token_id == -1:
                    continue

                pair = (doc, token_id)

                if pair not in pairs:
                    # document.append(doc)
                    # token.append(token_id)
                    # label.append(label_pred)
                    # token_str.append(tokens[token_id])
                    processed.append({"document": doc,
                                      "token": token_id,
                                      "label": label_pred,
                                      "token_str": tokens[token_id]})
                    pairs.append(pair)
        from spacy.lang.en import English
        nlp = English()

        def find_span(target: list[str],
                      document: list[str]) -> list[list[int]]:
            idx = 0
            spans = []
            span = []

            for i, token in enumerate(document):
                if token != target[idx]:
                    idx = 0
                    span = []
                    continue
                span.append(i)
                idx += 1
                if idx == len(target):
                    spans.append(span)
                    span = []
                    idx = 0
                    continue

            return spans
        test_data_for_regex_email_phone, _ = read_test(
            config.root_data_dir  # type: ignore
        )
        email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        phone_num_regex = re.compile(
            r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
        emails = []
        phone_nums = []

        for _data in test_data_for_regex_email_phone:
            # email
            for token_idx, token in enumerate(_data["tokens"]):
                if re.fullmatch(email_regex, token) is not None:
                    emails.append(
                        {
                            "document": _data["document"],
                            "token": token_idx,
                            "label": "B-EMAIL",
                            "token_str": token})
            # phone number
            matches = phone_num_regex.findall(_data["full_text"])
            if not matches:
                continue
            for match in matches:
                target = [t.text for t in nlp.tokenizer(match)]
                matched_spans = find_span(target, _data["tokens"])

            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    phone_nums.append(
                        {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM",
                         "token_str": _data["tokens"][token_idx]}
                    )

    df = pd.DataFrame(processed + phone_nums + emails)
    # Assign each row a unique 'row_id'
    df["row_id"] = list(range(len(df)))

    # Display a glimpse of the first 100 rows of your data
    # display(df.head(100))

    # Cast your findings into a CSV file for further exploration
    df[["row_id", "document", "token", "label"]].to_csv(
        config.inference_stage_1.output_path, index=False)


if __name__ == "__main__":
    seed_everything(seed=42)
    infer()
