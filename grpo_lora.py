import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import argparse

"""
功能：用于进行grpo训练

env:nobug
CUDA_VISIBLE_DEVICES=0 accelerate launch detoxllm/grpo_lora.py \
-m ./model_and_adpter/sft_model \
-o ./model_and_adpter/grpo_lora \
-s ./model_and_adpter/grpo_adapter \
-t 2.0 \
-a 5 \
-d ./data/new_grpo_train_para_data.json
"""

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="...")

# 添加参数
parser.add_argument("-m", "--modelpath", type=str, help="模型路径",default="/root/model2")
parser.add_argument("-o", "--outputdir", type=str, help="模型检查点保存路径") # "/root/detoxllm/grpo_lora_new_a10"
parser.add_argument("-s", "--savemodel", type=str, help="最终模型保存位置") # "/root/detoxllm/grpo_lora_new_a10_model"
parser.add_argument("-d", "--data", type=str, help="训练数据",default="/root/detoxllm/grpo_train_para_data.json")
parser.add_argument("-t", "--temp", type=float, help="生成温度",default=0.9)
parser.add_argument("-a", "--weighta", type=int, help="毒性权重",default=1)
parser.add_argument("-c", "--checkpointdir", type=str, help="检查点位置",default="/root/detoxllm/grpo_lora_new_a10/checkpoint-100000")
parser.add_argument("-e", "--epoch", type=int, help="训练轮次",default=1)

# 解析参数
args = parser.parse_args()

print("\n当前使用的参数配置：")
print("{:<20} {:<30} {:<50}".format("参数名称", "参数值", "说明"))
print("-" * 100)
print("{:<20} {:<30} {:<50}".format("--modelpath (-m)", args.modelpath, "模型路径"))
print("{:<20} {:<30} {:<50}".format("--outputdir (-o)", args.outputdir or "未设置", "模型检查点保存路径"))
print("{:<20} {:<30} {:<50}".format("--savemodel (-s)", args.savemodel or "未设置", "最终模型保存位置"))
print("{:<20} {:<30} {:<50}".format("--data (-d)", args.data, "训练数据"))
print("{:<20} {:<30} {:<50}".format("--temp (-t)", args.temp, "生成温度"))
print("{:<20} {:<30} {:<50}".format("--weighta (-a)", args.weighta, "毒性权重"))
print("{:<20} {:<30} {:<50}".format("--epoch (-e)", args.epoch, "训练轮次"))
print("{:<20} {:<30} {:<50}".format("--checkpointdir (-c)", args.checkpointdir, "检查点位置"))
print("-" * 100)


# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU. This may be slow.")


# LoRA 配置
lora_config = LoraConfig(
    r=8,  # LoRA 低秩矩阵维度（控制参数量）
    lora_alpha=16,  # LoRA 缩放因子
    lora_dropout=0.05,  # Dropout 防止过拟合
    bias="none",  # 不对 bias 进行 LoRA 适配
    target_modules=["q_proj","v_proj"],
)



# 加载基础模型和 tokenizer
model_name = args.modelpath
print(f"Loading model from {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Model and tokenizer loaded successfully.")

# 将模型移动到 GPU
model = model.to(device)

# 将模型转换为 LoRA 形式
print("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)
print("LoRA adaptation applied.")

# 检查可训练参数
model.print_trainable_parameters()

# 奖励函数
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from toxic_bert import BertMLPClassifier,predict
from transformers import BertTokenizer

# 加载模型
print("Loading embedding and detoxification models...")
embedder = SentenceTransformer("./model_and_adpter/all-MiniLM-L6-v2")  # 计算余弦相似度

toxic_model = BertMLPClassifier()
toxic_model.load_state_dict(torch.load("./model_and_adpter/toxic_bert/toxic_bert.pth",weights_only=True))# 检测毒性
toxic_model.eval()  # 设置为评估模式（不进行梯度更新）

toxic_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Models loaded successfully.")

def reward(completions, problem, **kwargs):
    """
    计算完成结果的语义相似度与毒性奖励。
    - 计算答案与标准答案的余弦相似度
    - 计算毒性评分，并对低毒性答案进行奖励
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, problem):
        # print(f"Evaluating: {content[:50]}...")  # 打印前 50 个字符进行检查

        # 计算语义相似度
        content_embedding = embedder.encode(content, convert_to_tensor=True)
        solution_embedding = embedder.encode(sol, convert_to_tensor=True)
        similarity = cosine_similarity([content_embedding.cpu().numpy()], [solution_embedding.cpu().numpy()])[0][0]
        # print(f"Cosine Similarity: {similarity}")

        # 计算毒性评分(毒标签为0 无毒为1) toxicity_score,non_toxic_score 分别对应两个标签的logits
        _,non_toxic_score = predict(content, toxic_model, toxic_tokenizer)

        # 计算最终奖励 (余弦相似度 + a * 毒性奖励)
        reward = similarity + args.weighta * non_toxic_score
        # print(toxicity_score,non_toxic_score,reward)
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

print("\n处理后的数据集样例：")
print("第一条原始数据样本：")
print(next(iter(dataset.values()))[0])

single_dataset = next(iter(dataset.values()))
train_test_split = single_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(f"len of train dataset:{len(train_dataset)}")

config = GRPOConfig(
    learning_rate=1e-5,  # LoRA 通常使用较小的学习率
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
    #调高温度使回答更富创意（defult=0.9）
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

checkpoint_dir = args.checkpointdir  # 替换为实际的检查点目录

if os.path.exists(checkpoint_dir):
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()
print("Training complete.")

# 保存 LoRA 适配的模型
save_path = args.savemodel
print(f"Saving model to {save_path}...")
trainer.save_model(save_path)
print("Model saved successfully.")