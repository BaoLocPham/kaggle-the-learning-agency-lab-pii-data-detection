import json
import re
from dataset.utils import read_test
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.convert_graph_to_onnx import convert
from onnxconverter_common import auto_convert_mixed_precision_model_path
import onnx
import torch.onnx
import onnxruntime
from utils import (
    get_logger,
    seed_everything,
    predict_and_convert,
    predict_and_quant,
    predict,
    process_predictions_ans,
)
from omegaconf import OmegaConf
import hydra
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import os
import gc
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")


LOGGER = get_logger(filename="inference_stage_1", logger_name="InferenceLogger")


class CFG:
    root_data_dir = "./data"
    output_dir = "./output"
    output_submission_path = "./"
    debug = True


config = CFG()


def tokenize_row(example, tokenizer, config):
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
        return_offsets_mapping=True,
        truncation=config.truncation,
        max_length=config.max_length,
    )

    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "offset_mapping": tokenized.offset_mapping,
        "token_map": token_map,
    }


def post_process_predicts(processed_predictions, ds, id2label):
    triplets = []
    pairs = (
        set()
    )  # membership operation using set is faster O(1) than that of list O(n)
    processed = []
    emails = []
    phone_nums = []
    urls = []
    streets = []
    LOGGER.info(id2label)
    # For each prediction, token mapping, offsets, tokens, and document in the
    # dataset
    for p, token_map, offsets, tokens, doc, full_text in zip(
        processed_predictions,
        ds["token_map"],
        ds["offset_mapping"],
        ds["tokens"],
        ds["document"],
        ds["full_text"],
    ):

        # Iterate through each token prediction and its corresponding offsets
        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[token_pred]  # Predicted label from token
            if start_idx + end_idx == 0:
                continue
            if token_map[start_idx] == -1:
                start_idx += 1
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1
            if start_idx >= len(token_map):
                break
            token_id = token_map[start_idx]  # Token ID at start index
            if (
                label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM")
                or token_id == -1
            ):
                continue
            pair = (doc, token_id)
            if pair not in pairs:
                processed.append(
                    {
                        "document": doc,
                        "token": token_id,
                        "label": label_pred,
                        "token_str": tokens[token_id],
                    }
                )
                pairs.add(pair)

        # email
        for token_idx, token in enumerate(tokens):
            if re.fullmatch(email_regex, token) is not None:
                emails.append(
                    {
                        "document": doc,
                        "token": token_idx,
                        "label": "B-EMAIL",
                        "token_str": token,
                    }
                )

        # phone number
        matches = phone_num_regex.findall(full_text)
        if not matches:
            continue
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, tokens)
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                phone_nums.append(
                    {
                        "document": doc,
                        "token": token_idx,
                        "label": f"{prefix}-PHONE_NUM",
                        "token_str": tokens[token_idx],
                    }
                )
        # url
        matches = url_regex.findall(full_text)
        if not matches:
            continue
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, tokens)
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                urls.append(
                    {
                        "document": doc,
                        "token": token_idx,
                        "label": f"{prefix}-URL_PERSONAL",
                        "token_str": tokens[token_idx],
                    }
                )
    return processed, phone_nums, emails, urls


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer(cfg):
    LOGGER.info(f"{OmegaConf.to_yaml(cfg)}")
    config.__dict__.update(cfg.parameters)

    # If it doesn't exist, create the folder
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.tmp_dir, exist_ok=True)
    test, submission = read_test(config.root_data_dir)  # type: ignore

    LOGGER.info(f"DEBUG MODE: {config.debug}")

    model_config = json.load(
        open(Path(list(config.inference_with_onnx.model_path.keys())[0], "config.json"))
    )
    id2label = model_config["id2label"]
    LOGGER.info(f"id2label: {id2label}")

    tokenizer = AutoTokenizer.from_pretrained(
        list(config.inference_with_onnx.model_path.keys())[0]
    )

    keep_cols = {"input_ids", "attention_mask"}
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=512)

    # test_ds = load_from_disk(f'{config.save_dir}test.dataset')
    df_test = pd.DataFrame(test)
    ds = Dataset.from_pandas(df_test)
    ds = ds.map(
        tokenize_row,
        fn_kwargs={"tokenizer": tokenizer, "config": config.inference_with_onnx},
        batched=False,
        num_proc=2,
        desc="Tokenizing",
    )
    test_ds = ds
    test_ds = test_ds.remove_columns(
        [c for c in test_ds.column_names if c not in keep_cols]
    )
    config.data_length = len(test_ds)
    config.len_token = len(tokenizer)
    LOGGER.info("Test Dataset created....")
    LOGGER.info((test_ds[0].keys()))
    LOGGER.info("Generating Test DataLoader")
    test_dataloader = DataLoader(
        test_ds,
        batch_size=config.inference_with_onnx.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collator,
    )

    # All predict data
    predictions_softmax_logits = []
    all_preds = []

    for model_path, weight in config.inference_with_onnx.model_path.items():

        # fold = config.trn_fold

        if config.inference_with_onnx.convert_before_inference:

            # Loading the original model and converting it to ONNX
            model = AutoModelForTokenClassification.from_pretrained(model_path)

            # Converting it to ONNX to a temp folder
            converted_model_name = (
                config.inference_with_onnx.temp_data_folder + "original_model.onnx"
            )
            predictions_softmax_all = predict_and_convert(
                test_dataloader, model, config, converted_model_name
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # In commit mode, save all quantized models with different names to create a dataset and reuse them later bypassing
            # vquantization and conversion
            quantized_model_name = (
                f"{config.inference_with_onnx.quantizated_model_dir}/optimized"
                + model_path.split("/")[-1]
                + ".onnx"
            )
            # data path should be relative
            quantized_data_path = "optimized" + model_path.split("/")[-1] + ".data"

            # Quantization
            predictions_softmax_all = predict_and_quant(
                test_dataloader,
                config,
                converted_model_name,
                quantized_model_name,
                quantized_data_path,
            )

        else:

            # Use already converted models, you can make a commit notebook once and save output models to a dataset,
            # for example, /kaggle/input/toonnx2-converted-models
            quantized_model_name = os.path.join(
                config.inference_with_onnx.quantizated_model_dir,
                os.path.basename(model_path),
                os.path.basename(model_path) + ".onnx",
            )
            LOGGER.info(f"Loading quantized model name: {quantized_model_name}")
        # Inference with ONNX
        LOGGER.info("Inference")

        # Create ONNX Runtime session for GPU
        session = onnxruntime.InferenceSession(
            quantized_model_name, providers=["CUDAExecutionProvider"]
        )
        # Uncomment this if you want to debug something on CPU
        # session = onnxruntime.InferenceSession(quantized_model_name)

        # Predict
        predictions_softmax_all = predict(test_dataloader, session, config)

        # Keep all logits for ensemble later
        predictions_softmax_logits.append(predictions_softmax_all)

    del test_dataloader, test_ds
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize an empty list to store the mean of the softmax predictions
    # from all models.
    predictions_mean_all = []

    # Calculate the total weight of all models to normalize the weights if its
    # sum exceeds 1.
    total_weight = sum(config.inference_with_onnx.model_paths.values())
    LOGGER.info(f"Total weight: {total_weight}")

    # Retrieve the individual weights for each model.
    model_weights = list(config.inference_with_onnx.model_paths.values())

    # Iterate over each sample since the length of texts can vary.
    for sample_index in range(len(predictions_softmax_logits[0])):

        # Initialize a tensor to accumulate weighted predictions for the
        # current sample.
        weighted_predictions_sum = torch.zeros(
            predictions_softmax_logits[0][sample_index].size()
        )

        # Iterate over each model to compute its contribution to the final
        # prediction.
        for model_index in range(len(predictions_softmax_logits)):
            weighted_prediction = predictions_softmax_logits[model_index][
                sample_index
            ] * (model_weights[model_index] / total_weight)
            weighted_predictions_sum += weighted_prediction

        # Append the mean of the weighted predictions for the current sample to
        # the list.
        predictions_mean_all.append(weighted_predictions_sum)

    processed, phone_nums, emails, urls = post_process_predicts(
        processed_predictions=processed_predictions, ds=ds, id2label=id2label
    )

    df = pd.DataFrame(processed + phone_nums + emails + urls)

    # Assign each row a unique 'row_id'
    df["row_id"] = list(range(len(df)))
    # Cast your findings into a CSV file for further exploration
    df[["row_id", "document", "token", "label"]].to_csv(
        config.inference_with_onnx.output_path, index=False
    )


if __name__ == "__main__":
    seed_everything(seed=42)
    infer()
