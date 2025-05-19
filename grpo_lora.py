import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import argparse

"""

env:nobug
CUDA_VISIBLE_DEVICES=0 accelerate launch detoxllm/grpo_lora.py \
-m ./model_and_adpter/sft_model \
-o ./model_and_adpter/grpo_lora \
-s ./model_and_adpter/grpo_adapter \
-t 2.0 \
-a 5 \
-d ./data/new_grpo_train_para_data.json
"""

parser = argparse.ArgumentParser(description="...")

parser.add_argument("-m", "--modelpath", type=str, help="Path to the model", default="/root/model2")
parser.add_argument("-o", "--outputdir", type=str, help="Directory to save model checkpoints")  # "/root/detoxllm/grpo_lora_new_a10"
parser.add_argument("-s", "--savemodel", type=str, help="Path to save the final model")  # "/root/detoxllm/grpo_lora_new_a10_model"
parser.add_argument("-d", "--data", type=str, help="Training data", default="/root/detoxllm/grpo_train_para_data.json")
parser.add_argument("-t", "--temp", type=float, help="Generation temperature", default=0.9)
parser.add_argument("-a", "--weighta", type=int, help="Toxicity weight", default=1)
parser.add_argument("-c", "--checkpointdir", type=str, help="Checkpoint directory", default="/root/detoxllm/grpo_lora_new_a10/checkpoint-100000")
parser.add_argument("-e", "--epoch", type=int, help="Number of training epochs", default=1)

args = parser.parse_args()

print("\nCurrent Configuration:")
print("{:<20} {:<30} {:<50}".format("Argument", "Value", "Description"))
print("-" * 100)
print("{:<20} {:<30} {:<50}".format("--modelpath (-m)", args.modelpath, "Path to the model"))
print("{:<20} {:<30} {:<50}".format("--outputdir (-o)", args.outputdir or "Not set", "Directory to save model checkpoints"))
print("{:<20} {:<30} {:<50}".format("--savemodel (-s)", args.savemodel or "Not set", "Path to save the final model"))
print("{:<20} {:<30} {:<50}".format("--data (-d)", args.data, "Training data"))
print("{:<20} {:<30} {:<50}".format("--temp (-t)", args.temp, "Generation temperature"))
print("{:<20} {:<30} {:<50}".format("--weighta (-a)", args.weighta, "Toxicity weight"))
print("{:<20} {:<30} {:<50}".format("--epoch (-e)", args.epoch, "Number of training epochs"))
print("{:<20} {:<30} {:<50}".format("--checkpointdir (-c)", args.checkpointdir, "Checkpoint directory"))
print("-" * 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU. This may be slow.")


# LoRA config
lora_config = LoraConfig(
    r=8,  
    lora_alpha=16,  
    lora_dropout=0.05,  
    bias="none",  
    target_modules=["q_proj","v_proj"],
)



model_name = args.modelpath
print(f"Loading model from {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Model and tokenizer loaded successfully.")


model = model.to(device)


print("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)
print("LoRA adaptation applied.")

model.print_trainable_parameters()

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from toxic_bert import BertMLPClassifier,predict
from transformers import BertTokenizer

print("Loading embedding and detoxification models...")
embedder = SentenceTransformer("./model_and_adpter/all-MiniLM-L6-v2") 

toxic_model = BertMLPClassifier()
toxic_model.load_state_dict(torch.load("./model_and_adpter/toxic_bert/toxic_bert.pth",weights_only=True))
toxic_model.eval()  

toxic_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Models loaded successfully.")

def reward(completions, problem, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, problem):

        content_embedding = embedder.encode(content, convert_to_tensor=True)
        solution_embedding = embedder.encode(sol, convert_to_tensor=True)
        similarity = cosine_similarity([content_embedding.cpu().numpy()], [solution_embedding.cpu().numpy()])[0][0]
        _,non_toxic_score = predict(content, toxic_model, toxic_tokenizer)


        reward = similarity + args.weighta * non_toxic_score
        rewards.append(reward)

    return rewards

SYSTEM_PROMPT = (
    "You are a text detoxification assistant. Your task is to detoxify the given text and provide a non-toxic, "
    "non-offensive, non-discriminatory, and safe response while preserving the original meaning."
    "Do not include unnecessary reasoning or explanations, only provide the final answer."
)
PROMPT = (
    "Please follow these steps:1.Identify and highlight the toxic words or phrases in the text."
    "2.Replace the identified toxic terms with neutral alternatives, ensuring the sentence's structure, tone, and meaning remain unchanged."
    "3.Ensure that the revised sentence is entirely non-toxic, while maintaining the same viewpoint and stance as the original."
    "Please provide the revised, non-toxic sentence."
)

# Load datasets
from datasets import load_dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=args.data)
print("Dataset loaded.")

# Prepare dataset
print("Processing dataset...")
dataset = dataset.map(
    lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': PROMPT+x['problem']}
        ],
        'answer': x['solution']
    },
    desc="Processing dataset",
)
print("Dataset processed.")

print("\nSample from the processed dataset:")
print("First raw data sample:")
print(next(iter(dataset.values()))[0])

single_dataset = next(iter(dataset.values()))
train_test_split = single_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(f"len of train dataset:{len(train_dataset)}")

config = GRPOConfig(
    learning_rate=1e-5,  
    eval_strategy="steps",
    eval_steps=200,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 50,
    bf16 = True,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8, 
    num_generations = 4, 
    max_prompt_length = 256,
    max_completion_length = 300,
    num_train_epochs=args.epoch,
    save_steps = 1073,
    max_grad_norm = 1.0,
    report_to = "tensorboard",
    output_dir = args.outputdir,
    temperature=args.temp
)

print("Initializing trainer...")

import os
from peft import PeftModel


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    reward_funcs=reward,
    peft_config=lora_config,
)

print("Trainer initialized.")

print("Starting training...")

checkpoint_dir = args.checkpointdir  

if os.path.exists(checkpoint_dir):
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()
print("Training complete.")

save_path = args.savemodel
print(f"Saving model to {save_path}...")
trainer.save_model(save_path)
print("Model saved successfully.")