---
library_name: peft
license: other
base_model: /root/shared_planing/LLM_model/Llama3-8B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: new_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# new_sft

This model is a fine-tuned version of [/root/shared_planing/LLM_model/Llama3-8B-Instruct](https://huggingface.co//root/shared_planing/LLM_model/Llama3-8B-Instruct) on the alpaca_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8501

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.5071        | 0.8889 | 50   | 0.9277          |
| 0.7954        | 1.7644 | 100  | 0.8573          |
| 0.6988        | 2.64   | 150  | 0.8507          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0