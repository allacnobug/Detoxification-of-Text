# Text Detoxification: Data Efficiency, Semantic Preservation and Model Generalization

This project builds a text detoxification system based on the LLaMA3 model, aiming to improve semantic consistency and model robustness. The system supports two usage modes:

- **Chat Mode for Interactive Dialogue**: Users can directly interact with the model to perform personalized detoxification tasks.
- **Standardized API/Interface Mode**: Supports batch processing of large volumes of toxic texts.

---

## How to Use the Text Detoxification System

### Step 1: Environment Setup

You need to prepare the following:

1. *(Optional)* Create a new Python virtual environment for use with LLaMA-Factory.
2. Install third-party dependencies required by LLaMA-Factory (via `requirements.txt`).
3. Install the LLaMA-Factory core and ensure that the `llamafactory-cli` is generated.

#### Reference Commands

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
```

For detailed configuration, please refer to the [documentation](https://zhuanlan.zhihu.com/p/695287607).

You can also directly use the provided `llama_factory.yml` file for environment deployment.

---

### Step 2: Model Preparation

Download [LLaMA3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) to the local `model_and_adapter` folder.

Run the following code to merge the model (within the `llama_factory` environment):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model path configuration
base_model_name = "model_and_adpter/Llama3.1-8B-Instruct"
adapter_model_name = "model_and_adpter/sft_adapter"
output_dir = "model_and_adpter/sft_model"

# Load base model and tokenizer
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)

# Load adapter
print("Merging adapter...")
model = PeftModel.from_pretrained(model, adapter_model_name)

# Merge model parameters
print("Merging model weights...")
model = model.merge_and_unload()

# Save the full model
print(f"Saving full model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Done!")
```
### Step 3: Start Detoxification

#### Method 1: Chat Mode

After downloading the model and adapter and verifying the paths, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat \
--model_name_or_path model_and_adpter/sft_model \
--adapter_name_or_path model_and_adpter/grpo_adapter \
--template llama3 \
--finetuning_type lora
```
When using the system, please refer to the detox instruction we provide, and replace the content inside the brackets `[...]` with the sentence you want to detox:
```text
Please follow these steps: 1.Remove or replace with neutral terms all toxic content in this sentence, including attacks, biases, discrimination, insults, hatred, pornography, threats, intimidation, derogatory language, politically sensitive material, or impolite expressions. 2.Delete or rephrase any derogatory terms and disrespectful language. 3.Note: Identify ​all toxic elements in the sentence, which may occur in multiple instances. 4.The rewritten sentence must preserve the original meaning with structurally and tonally similar phrasing. 5.Output only the revised sentence without explanations. Now Detoxify the following sentence, ensuring it contains no harmful content while preserving the original viewpoint and emotional tone:+[toxic sentence]
```
**You can also use the web chat mode by simply replacing `chat` with `webchat`:**

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
--model_name_or_path model_and_adapter/sft_model \
--adapter_name_or_path model_and_adapter/grpo_adapter \
--template llama3 \
--finetuning_type lora
```

#### Method 2: Batch Inference

First, prepare a CSV file containing a `toxic` column, for example, `paradetox_test_671.csv`, then perform data format conversion:

```python
import pandas as pd
import json

df = pd.read_csv('data/paradetox_test_671.csv')
alpaca_data = []
for _, row in df.iterrows():
    data = {
        "instruction":"Please follow these steps:1.Identify and highlight the toxic words or phrases in the text.2.Replace the identified toxic terms with neutral alternatives, ensuring the sentence's structure, tone, and meaning remain unchanged.3.Ensure that the revised sentence is entirely non-toxic, while maintaining the same viewpoint and stance as the original.Please provide the revised, non-toxic sentence.",
        "input": row['toxic']if 'toxic' in row else "",
        "output": row['neutral1'] if 'neutral1' in row else "",
        "system": "You are a text de-toxification system. Your task is to convert the following toxic text into a non-toxic version while preserving the original meaning and tone.",
    }
    alpaca_data.append(data)

with open('./data/paratest_671.json', 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=4)
```
Place the result file into `LLaMA-Factory/data` and modify `dataset_info.json`. Then you can use the following command for batch detoxification:

```bash
CUDA_VISIBLE_DEVICES=0 python LLaMA-Factory/scripts/vllm_infer.py \
--model_name_or_path sft_model \
--adapter_name_or_path adapter/grpo_adapter \
--dataset paratest \
--dataset_dir LLaMA-Factory/data \
--template llama3 \
--save_name paratest_generated_predictions.jsonl
```
You can use the following script for formatting:
```bash
python clean_jsonl.py -f paratest
```

---
## How to Reproduce This Project

### Step 1: Data Preparation

Same as batch inference, but training requires a parallel dataset (you can use the provided `sft_train_para_data.json`).

### Step 2: Cold Start

First, download the LLaMA3-8B-Instruct base model, then run:
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
--stage sft \
--do_train \
--model_name_or_path model_and_adpter/Llama3-8B-Instruct \
--dataset sft_train \
--dataset_dir ./LLaMA-Factory/data \
--template llama3 \
--finetuning_type lora \
--output_dir model_and_adapter/sft_adapter \
--overwrite_cache \
--overwrite_output_dir \
--cutoff_len 1024 \
--preprocessing_num_workers 16 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 50 \
--warmup_steps 20 \
--save_steps 100 \
--eval_steps 25 \
--evaluation_strategy steps \
--load_best_model_at_end \
--learning_rate 5e-5 \
--num_train_epochs 3.0 \
--max_samples 1000 \
--val_size 0.1 \
--plot_loss \
--fp16
```
#### Merge Model

```bash
CUDA_VISIBLE_DEVICES=7 llamafactory-cli export \
--model_name_or_path model_and_adpter/Llama3-8B-Instruct \
--adapter_name_or_path model_and_adpter/sft_adapter \
--template llama3 \
--finetuning_type lora \
--export_dir model_and_adpter/sft_model \
--export_size 2 \
--export_device cpu \
--export_legacy_format False
```

---

### Step 3: GRPO Reinforcement Learning

**Create a new virtual environment named `nobug`, and you can install dependencies using the provided YAML file.**

Download the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.

Use `toxic_bert.py` to train a toxicity classifier and save the trained model to `model_and_adpter/toxic_bert`.

Use the following code to perform GRPO training:

```bash
conda activate nobug

CUDA_VISIBLE_DEVICES=0 accelerate launch grpo_lora.py \
-m model_and_adpter/sft_model \
-o model_and_adpter/grpo_lora \
-s model_and_adpter/grpo_adapter \
-t 2.0 \
-a 5 \
-d data/new_grpo_train_para_data.json
```

---
### Step 4: Model Evaluation

Integrate evaluation metrics from ParadeTox:
- **STA**: Semantic Preservation
- **SIM**: Semantic Similarity
- **FL**: Fluency
- **J**: Toxicity Classification Accuracy

First, perform batch inference:
```bash
CUDA_VISIBLE_DEVICES=3 python LLaMA-Factory/scripts/vllm_infer.py \
--model_name_or_path model_and_adapter/sft_model \
--adapter_name_or_path model_and_adapter/grpo_adapter \
--dataset paratest \
--dataset_dir LLaMA-Factory/data \
--template llama3 \
--save_name paratest_generated_predictions.jsonl
```

Download the [wieting similarity](https://storage.yandexcloud.net/nlp/wieting_similarity_data.zip) model to the `evaluation_metric` folder.

Download the [CoLA classifier](https://drive.google.com/drive/folders/1p6_3lCbw3J0MhlidvKkRbG73qwmtWuRp) model to the `evaluation_metric` folder.

Then perform data cleaning and scoring:
```bash
python clean_jsonl.py -f paratest
python evaluation_detox/metric.py -i paratest_generated_predictions.csv
```

The results will be saved to `metric_results.md`.

---

## Appendix

- Explanation of the `data` folder:
    - `dataset_info.json`: An example format of LLaMA-Factory/data/dataset_info.json
    - `detoxtest.json`: Test set filtered from DetoxLLM
    - `hugtest.json`: Dataset from the Huggingface platform
    - `paratest_671.json`: ParaDetox test set
    - `sft_train_para_data.json`: Dataset used for cold start
    - `para_data_classify.csv`: Dataset used for training the toxicity classifier
    - `paradetox_test_671.csv`: Test set file
    - `train_para_data.csv`: Training set file
    - `new_grpo_train_para_data.json`: GRPO training file

---

## References and Dataset Sources

[Paradetox](https://github.com/s-nlp/paradetox/tree/main)

[DetoxLLM](https://huggingface.co/UBC-NLP/DetoxLLM-7B)

[Huggingface Dataset](https://huggingface.co/datasets/narySt/text_detoxification_dataset)

[You Only Prompt Once](https://github.com/xinleihe/toxic-prompt#)