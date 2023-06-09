# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""


import dataclasses
import logging
import os
import sys

sys.path.insert(0, '/home/harduin/Desktop/Research/CMSC848D_Project/Spurious_Correlations-20230425T172048Z-001/Spurious_Correlations/Identifying-and-Mitigating-Spurious-Correlations-for-Improving-Robustness-in-NLP-Models/utils/')

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from trainer import Trainer

import torch
from torch.utils.data.dataloader import DataLoader

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    PretrainedConfig,  
    DataCollatorWithPadding,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import datasets
from datasets import load_dataset
logger = logging.getLogger(__name__)

from data_structure import get_yelp, get_sst2, get_movie_rationales

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))


    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst-2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=data_args.task_name,
    #     cache_dir=model_args.cache_dir,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    # )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # use_fast=model_args.use_fast_tokenizer,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    #tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    #prajjwal1/bert-small-mnli, ishan/bert-base-uncased-mnli, barissayil/bert-sentiment-analysis-sst
    #model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    #model = AutoModelWithLMHead.from_pretrained("barissayil/bert-sentiment-analysis-sst")

    # Get datasets
    # train_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    # )
    # eval_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    #     if training_args.do_eval
    #     else None
    # )
    # test_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
    #     if training_args.do_predict
    #     else None
    # )
    
    # HG OVERWRITE
    # Overwrite the train/eval/test datasets with up to date methods for extracting datasets
    # import pandas as pd
    # print("Is this a valid dataloader argumetn?", training_args.per_device_train_batch_size)
    # raw_datasets = pd.read_table(os.path.join(data_args.data_dir, "train.tsv"))
    
    # train_X = torch.tensor(train_df['sentence']).values.astype(dtype_string)
    # train_y = torch.tensor(train_df['label']).values.astype(np.float32)
    # train_tensor = data_utils.TensorDataset(train_X, train_y)
    # train_dataset = DataLoader(train_tensor, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    # test_dataset = pd.read_table(os.path.join(data_args.data_dir, "test.tsv"))
    # eval_dataset = pd.read_table(os.path.join(data_args.data_dir, "dev.tsv"))
    avail_datasets = ["sst2", "yelp", "movie_rationales"] 
    data_idx = 2
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        # Uncomment below to load sst2 dataset

        if avail_datasets[data_idx] == "sst2":
            # raw_datasets = load_dataset(
            #     "glue",
            #     'sst2',# HG HARD SETTING THIS FOR NOW
            #     cache_dir=model_args.cache_dir,
            #     # use_auth_token=True if model_args.use_auth_token else None,
            # )

            train_df = get_sst2()
            num_from_split = 7500
        elif avail_datasets[data_idx] == "yelp":
            # raw_datasets = load_dataset("yelp_polarity")
            train_df = get_yelp()
            num_from_split = 7500
        elif avail_datasets[data_idx] == "movie_rationales":
            train_df = get_movie_rationales()
            num_from_split = 500
            
    num_sen = len(train_df)

    train_split_1 = 0
    train_split_2 = num_sen // 3
    train_split_3 = num_sen // 3 * 2

    new_train_df = train_df[train_split_1:train_split_1 +num_from_split] 
    new_train_df = new_train_df.append(train_df[train_split_2:train_split_2 +num_from_split] )
    new_train_df = new_train_df.append( train_df[train_split_3:train_split_3 +num_from_split] )

    train_df = new_train_df.reset_index()
    raw_datasets = datasets.Dataset.from_pandas(train_df)
    print("glue data set is: ", raw_datasets)

       # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    # if data_args.pad_to_max_length: # HG OVERWRITE, need to check if padding is enabled later
    #     padding = "max_length"
    # else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = "max_length"

    is_regression = data_args.task_name == "stsb" # Definitely not true because only doing sentiment classification
    # label_list = raw_datasets["train"].features["label"].names
    label_list = ['negative', 'positive']
    num_labels = len(label_list)

    print("Label list looks like: ", label_list)
    
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        # print(f'sentence1 key is : {sentence1_key}\n\n')
        # HG just hardcoding to text when using the Yelp Dataset
        sentence1_key = "text" #if yelp
        # sentence1_key = "sentence" # if sst2
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        # if "validation" not in raw_datasets and "validation_matched" not in raw_datasets and "test" not in raw_datasets:
            # raise ValueError("--do_eval requires a validation dataset")
   
        # eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"] # Normal way of doing it
        # eval_dataset = raw_datasets["train"] # HG way of doing it, just use test set for validation set since yelp polarity doesn't have a val set
        eval_dataset = raw_datasets # HG way of doing it, just use test set for validation set since yelp polarity doesn't have a val set
        
        # eval_dataset = raw_datasets["test"] # HG way of doing it, just use test set for validation set since yelp polarity doesn't have a val set
        print("len of eval dataset is: ", len(eval_dataset))
        # if data_args.max_eval_samples is not None:
        #     max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        #     eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        if True:
            max_eval_samples = len(eval_dataset)
            eval_dataset = eval_dataset.select(range(max_eval_samples))    

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        # if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        #     raise ValueError("--do_predict requires a test dataset")
        # predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        predict_dataset = raw_datasets
        # if data_args.max_predict_samples is not None:
        #     max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        #     predict_dataset = predict_dataset.select(range(max_predict_samples))
        if True:
            max_predict_samples = len(predict_dataset)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
      # END OF HG OVERWRITE
      
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # HG OVERWRITE
   # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    # if data_args.pad_to_max_length:
    #     data_collator = default_data_collator
    # elif training_args.fp16:
    #     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # else:
    #     data_collator = None
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    #END OF HG OVERWRITE

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        # collate_fn=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            # If we want to reduce the size of the yelp test dataset
            # print(len(eval_dataset))
            # if len(eval_dataset) > 8304:
            #     import torch
            #     indices = torch.arange(8304)
            #     eval_dataset = eval_dataset.select(indices)
            trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name)
            # print(eval_dataset)
            # import torch
            # indices = torch.arange(8)
            # new_eval_dataset = eval_dataset.select(indices)
            # print(new_eval_dataset)
            # print("I want to see the first sentence of new eval dataset")
            # print(new_eval_dataset[0])
            # print("_______")
            
            eval_result, attention_weights, logits = trainer.evaluate(eval_dataset=eval_dataset)
            print("We have successfully dumped attention weights or whatever")
            # dump attention weights in numpy
            with open(os.path.join(training_args.output_dir, "cls_attention_weights.npy"), "wb") as f:
                np.save(f, attention_weights.detach().cpu().numpy())
            with open(os.path.join(training_args.output_dir, "predictions.npy"), "wb") as f:
                np.save(f, logits[0])

            print("attention weights size: ", attention_weights.size())
            print("logits size", logits[0].shape)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{data_args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(data_args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
