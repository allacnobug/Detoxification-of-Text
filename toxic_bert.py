import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


maxlen = 256
batch_size = 16
epochs = 5
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=maxlen):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

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
            'input_ids': encoding['input_ids'].squeeze(0),  
            'attention_mask': encoding['attention_mask'].squeeze(0),  
            'label': torch.tensor(label, dtype=torch.long)  
        }


# BERT+MLP
class BertMLPClassifier(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', hidden_dim=128, num_classes=2):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)  

        
        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])  
                if layer_num < 10:  
                    param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  
        return self.fc(pooled_output)  
    
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device):

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.AdamW(model.parameters(), lr=lr)  
    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
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

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig('./result/loss_curve.png') 
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.savefig('./result/accuracy_curve.png')  
    plt.show()

    print("Done！")
    return model

def predict_class(text, model, tokenizer, max_len=256, device=device):

    model.to(device)
    model.eval()  
    
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class

def predict(text, model, tokenizer, max_len=256, device=device):

    model.to(device)
    model.eval() 
    
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    softmax_outputs = F.softmax(outputs, dim=1)
    toxic_score = softmax_outputs[0, 0].item()  
    non_toxic_score = softmax_outputs[0, 1].item()  

    return toxic_score,non_toxic_score

if __name__ == "__main__":
    df = pd.read_csv('data/para_data_classify.csv')  
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BertMLPClassifier()
    # model.load_state_dict(torch.load("/root/detoxllm/toxic_bert/toxic_bert.pth"))
    model.to(device)

    trained_model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    # trained_model = train_model(model, train_loader, val_loader, device=device)
    print("Done")

    model_path = 'model_and_adpter/toxic_bert/toxic_bert.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"model saved in {model_path}")

    # model_path = '/root/detoxllm/toxic_bert/toxic_bert.pth'
    # model.load_state_dict(torch.load(model_path))

