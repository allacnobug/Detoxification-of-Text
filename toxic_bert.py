import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""用来训练毒性分类器的代码"""

# 超参数
maxlen = 256
batch_size = 16
epochs = 5
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=maxlen):
        """
        初始化数据集
        :param texts: 文本数据列表
        :param labels: 标签列表
        :param tokenizer: BERT tokenizer
        :param max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        处理单个样本，进行分词和编码
        :param idx: 数据索引
        :return: 编码后的文本及其标签
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 词索引
            'attention_mask': encoding['attention_mask'].squeeze(0),  # 注意力掩码
            'label': torch.tensor(label, dtype=torch.long)  # 转换标签为Tensor
        }


# BERT+MLP分类器
class BertMLPClassifier(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', hidden_dim=128, num_classes=2):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)  # 加载预训练BERT

        # 冻结BERT的前几层
        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])  # 获取层数
                if layer_num < 10:  # 假设冻结前10层
                    param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # 取[CLS] token的向量
        return self.fc(pooled_output)  # 通过MLP进行分类
    
import torch.nn.functional as F

# 3. 训练和评估函数
def train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device):
    """
    训练模型
    :param model: BERT+MLP模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :param device: 计算设备（CPU/GPU）
    """
    # print("开始训练模型...")
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # AdamW优化器
    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"开始第 {epoch + 1} 轮训练...")
        model.train()
        total_loss, total_correct = 0, 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
                'label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_acc = total_correct / len(train_loader.dataset)
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
                    'label'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}: Loss {total_loss:.4f}, Train Accuracy {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Accuracy {val_acc:.4f}")

    # 绘制损失曲线并保存
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig('./result/loss_curve.png')  # 保存图片
    plt.show()

    # 绘制准确率曲线并保存
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.savefig('./result/accuracy_curve.png')  # 保存图片
    plt.show()

    print("训练完成！")
    return model

def predict_class(text, model, tokenizer, max_len=256, device=device):

    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 对输入文本进行分词和编码
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 进行前向传播，获取预测结果
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # 获取最高分对应的类别
    predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class

def predict(text, model, tokenizer, max_len=256, device=device):

    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 对输入文本进行分词和编码
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 进行前向传播，获取预测结果
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # 获取最高分对应的类别
    # predicted_class = torch.argmax(outputs, dim=1).item()
    # 应用 Softmax 函数
    softmax_outputs = F.softmax(outputs, dim=1)
    toxic_score = softmax_outputs[0, 0].item()  # 第一个类别的得分
    non_toxic_score = softmax_outputs[0, 1].item()  # 第二个类别的得分

    return toxic_score,non_toxic_score

# 4. 运行代码
if __name__ == "__main__":
    print("开始加载数据...")
    # 加载数据
    df = pd.read_csv('./data/para_data_classify.csv')  # 确保你的 CSV 文件包含 'text' 和 'label' 两列
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print("数据加载完成。")

    print("开始划分数据集...")
    # 划分数据集（90%训练，10%验证）
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    print("数据集划分完成。")

    print("开始初始化Tokenizer...")
    # 初始化Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer初始化完成。")

    print("开始创建DataLoader...")
    # 创建DataLoader
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print("DataLoader创建完成。")

    print("开始初始化模型...")
    model = BertMLPClassifier()
    model.load_state_dict(torch.load("/root/detoxllm/toxic_bert/toxic_bert.pth"))
    model.to(device)
    print("模型初始化完成。")

    print("开始训练模型...")
      # 继续训练模型
    # additional_epochs = 10  # 额外训练的轮数
    trained_model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    # trained_model = train_model(model, train_loader, val_loader, device=device)
    print("模型训练完成。")

    # 保存模型
    model_path = './model/toxic_bert/toxic_bert.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # 加载保存的模型参数
    # model_path = '/root/detoxllm/toxic_bert/toxic_bert.pth'
    # model.load_state_dict(torch.load(model_path))

