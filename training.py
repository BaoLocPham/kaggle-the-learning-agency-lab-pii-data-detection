from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
import json
import argparse
from itertools import chain
from functools import partial
from dataset.learning_agency_dataset import tokenize
from dataset.utils import read_data

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features
import numpy as np

from utils import get_logger, seed_everything
import wandb
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import random
import warnings
import os
import gc
import re
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")


LOGGER = get_logger(filename="training_stage_1", logger_name="TrainingLogger")


class CFG:
    root_data_dir = "./data"
    output_dir = "./output"
    output_submission_path = "./"
    debug = True


config = CFG()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def init_experiment(config):
    configs = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True)
    # debug = configs["parameters"]["debug"]
    configs = configs["parameters"]
    if configs["train_stage_1"]["wandb"]:
        wandb.login(key=os.environ["WANDB"])
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            group="training_stage_1",
            config=configs,
        )
    return wandb


def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall)

    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    LOGGER.info(f"{OmegaConf.to_yaml(cfg)}")
    config.__dict__.update(cfg.parameters)

    # if not os.path.exists(config.output_dir):
    # If it doesn't exist, create the folder
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.train_stage_1.output_dir, exist_ok=True)
    train, test, submission = read_data(
        config.root_data_dir, config=config, debug=config.debug  # type: ignore
    )

    LOGGER.info(f"DEBUG MODE: {config.debug}")
    LOGGER.info(f"Train shape: {train.shape}")

    all_labels = sorted(list(set(chain(*[x["labels"] for x in train]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    LOGGER.info(f"id2label: {id2label}")

    tokenizer = AutoTokenizer.from_pretrained(config.train_stage_1.model_path)

    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in train],
        "document": [x["document"] for x in train],
        "tokens": [x["tokens"] for x in train],
        "trailing_whitespace": [x["trailing_whitespace"] for x in train],
        "provided_labels": [x["labels"] for x in train],
    })

    ds = ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": config.train_stage_1.max_len,
            "config": config},
        num_proc=2)
    # ds = ds.class_encode_column("group")

    x = ds[0]

    for t, l in zip(x["tokens"], x["provided_labels"]):
        if l != "O":
            LOGGER.info(f"{(t,l)}")

    LOGGER.info("*" * 100)

    for t, l in zip(
            tokenizer.convert_ids_to_tokens(
            x["input_ids"]), x["labels"]):
        if id2label[l] != "O":
            LOGGER.info(f"{(t,id2label[l])}")

    # may want to try to balance classes in splits
    final_ds = ds.train_test_split(
        test_size=config.train_stage_1.test_size,
        seed=42)  # cannot use stratify_by_column='group'
    final_ds

    model = AutoModelForTokenClassification.from_pretrained(
        config.train_stage_1.model_path,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=16)

    config.train_stage_1.freeze_embeddings = False
    config.train_stage_1.freeze_n_layers = 6

    if config.train_stage_1.freeze_embeddings:
        LOGGER.info('Freezing embeddings.')
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False

    if config.train_stage_1.freeze_n_layers > 0:
        LOGGER.info(f'Freezing {config.train_stage_1.freeze_n_layers} layers.')
        for layer in model.deberta.encoder.layer[:
                                                 config.train_stage_1.freeze_n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    args = TrainingArguments(
        output_dir=config.train_stage_1.output_dir,
        fp16=config.train_stage_1.fp16,
        learning_rate=config.train_stage_1.learning_rate,
        num_train_epochs=config.train_stage_1.num_train_epochs,
        per_device_train_batch_size=config.train_stage_1.per_device_train_batch_size,
        per_device_eval_batch_size=config.train_stage_1.per_device_eval_batch_size,
        gradient_accumulation_steps=config.train_stage_1.gradient_accumulation_steps,
        report_to=config.train_stage_1.report_to,
        evaluation_strategy=config.train_stage_1.evaluation_strategy,
        do_eval=config.train_stage_1.do_eval,
        save_strategy=config.train_stage_1.save_strategy,
        save_total_limit=config.train_stage_1.save_total_limit,
        overwrite_output_dir=config.train_stage_1.overwrite_output_dir,
        load_best_model_at_end=config.train_stage_1.load_best_model_at_end,
        lr_scheduler_type=config.train_stage_1.lr_scheduler_type,
        metric_for_best_model=config.train_stage_1.metric_for_best_model,
        greater_is_better=config.train_stage_1.greater_is_better,
        warmup_ratio=config.train_stage_1.warmup_ratio,
        weight_decay=config.train_stage_1.weight_decay,
        logging_steps=config.train_stage_1.logging_steps
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=final_ds["train"],
        eval_dataset=final_ds["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, all_labels=all_labels),
    )
    trainer.train()
    trainer.save_model(config.train_stage_1.output_dir)
    tokenizer.save_pretrained(config.train_stage_1.output_dir)
    if config.train_stage_1.wandb:
        wandb.finish()


if __name__ == "__main__":
    seed_everything(seed=42)
    init_experiment()
    train()
