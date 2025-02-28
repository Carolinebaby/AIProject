import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt


class QNLIDataset(Dataset):
    def __init__(self, file_path, glove_embeddings_index, dim=100, max_length=128):
        self.max_length = max_length
        self.dim = dim
        self.glove_embeddings_index = glove_embeddings_index
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            f.readline()  # Skip the first line
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    question = parts[1].lower()
                    sentence = parts[2].lower()
                    label = 1 if parts[3] == 'entailment' else 0
                    data.append([question, sentence, label])
        return data

    def __getitem__(self, idx):
        question, sentence, label = self.data[idx]

        # Tokenize and create embeddings
        text = question + " " + sentence
        text_embedding = self.get_embedding(text)

        return {
            'text': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

    def get_embedding(self, text):
        tokens = text.split()
        embedding = [self.glove_embeddings_index.get(token, np.zeros(self.dim, dtype=np.float32))
                     for token in tokens]

        if len(embedding) < self.max_length:
            embedding = [np.zeros(self.dim, dtype=np.float32)] * (self.max_length - len(embedding)) + embedding
        else:
            embedding = embedding[:self.max_length]

        return np.array(embedding)


def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index


class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # hidden_size * 2 because of bidirectional
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # * 2 because of bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, eval_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in eval_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            outputs = model(text)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def main():
    start_time = time()
    train_path = './QNLI_all/train.tsv'          # 总的训练数据路径
    test_path = './QNLI_all/dev.tsv'             # 总的测试数据路径
    glove_path = './glove.6B/glove.6B.100d.txt'  # GloVe词向量路径
    max_length = 100      # 最大序列长度
    glove_dim = 100       # 词向量维度
    hidden_size = 128     # 隐藏层大小
    num_layers = 1        # LSTM层数
    output_size = 2       # 输出类别数
    num_epochs = 10       # 训练轮数
    batch_size = 16       # 批次大小
    lr = 1e-3             # 学习率
    dropout = 0.2         # dropout率


    glove_embeddings_index = load_glove_embeddings(glove_path)

    train_dataset = QNLIDataset(train_path, glove_embeddings_index, dim=glove_dim, max_length=max_length)
    test_dataset = QNLIDataset(test_path, glove_embeddings_index, dim=glove_dim, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTMNet(glove_dim, hidden_size, num_layers, output_size, dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # 记录训练过程中的准确率和损失值
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)  # 训练模型
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

        train_accuracy = evaluate(model, train_loader, device)  # 评估训练集准确率
        train_accuracies.append(train_accuracy)
        print(f'Train Accuracy: {train_accuracy:.4f}')

        test_accuracy = evaluate(model, test_loader, device)  # 评估测试集准确率
        test_accuracies.append(test_accuracy)
        print(f'Test Accuracy: {test_accuracy:.4f}')

        scheduler.step()  # 更新学习率

    end_time = time()
    print(f'Total Time: {end_time - start_time:.1f} s')  # 输出总时间

    # 绘制准确率和损失值变化折线图
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # 绘制训练集和测试集准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, '-o', label='Train Accuracy')
    plt.plot(epochs, test_accuracies, '-o', label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.legend()

    # 绘制训练损失值
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, '-o', label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
