---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: distilbert_SST_base_cased
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert_SST_base_cased

This model is a fine-tuned version of [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) on the GLUE SST2 dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.3902
- eval_accuracy: 0.9106
- eval_runtime: 13.3502
- eval_samples_per_second: 65.317
- eval_steps_per_second: 8.165
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1.0

### Framework versions

- Transformers 4.28.0.dev0
- Pytorch 1.13.1+cu116
- Datasets 2.10.1
- Tokenizers 0.13.2
