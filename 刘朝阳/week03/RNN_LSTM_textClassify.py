import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

random.seed(42)
torch.manual_seed(42)

# ============ 1. 数据准备 ============
CHARS = '阿斯顿你就是一二三四五六七七八九十些不了人我在有他这中大来上个国说们为子和你地而出道也时可得那下后自以会心可'
CHAR_LIST = list(CHARS)
NUM_CLASSES = 5
VOCAB_SIZE = len(CHAR_LIST) + 1

char2idx = {c: i+1 for i, c in enumerate(CHAR_LIST)}
char2idx['<UNK>'] = 0

# ============ 2. 数据集 ============
class TextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        indices = [char2idx.get(c, 0) for c in text]
        return torch.tensor(indices, dtype=torch.long), label - 1

# ============ 3. RNN模型 ============
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.rnn = nn.RNN(64, 64, batch_first=True)
        self.fc = nn.Linear(64, NUM_CLASSES)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# ============ 4. LSTM模型 ============
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, NUM_CLASSES)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# ============ 5. 生成数据集 ============
def generate_data(num_samples):
    samples = []
    for _ in range(num_samples):
        pos = random.randint(1, 5)
        text = ''
        for i in range(5):
            if i + 1 == pos:
                text += '你'
            else:
                text += random.choice(CHAR_LIST)
        samples.append((text, pos))
    return samples

# ============ 6. 训练模型 ============
def train_model(model, train_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={100*correct/total:.2f}%')

# ============ 7. 测试 ============
def test_model(model, test_samples):
    correct = 0
    for text, label in test_samples:
        indices = torch.tensor([[char2idx.get(c, 0) for c in text]], dtype=torch.long)
        with torch.no_grad():
            output = model(indices)
        _, predicted = output.max(1)
        if predicted.item() + 1 == label:
            correct += 1
        else:
            print(f'Mispred: text="{text}", true={label}, pred={predicted.item()+1}, output={output}')
    print(f'Test Accuracy: {100*correct/len(test_samples):.2f}%')

def main():
    train_samples = generate_data(2000)
    test_samples = generate_data(500)
    
    train_dataset = TextDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print('Training RNNModel...')
    rnn_model = RNNModel()
    train_model(rnn_model, train_loader)
    test_model(rnn_model, test_samples)
    
    print('\nTraining LSTMModel...')
    lstm_model = LSTMModel()
    train_model(lstm_model, train_loader)
    test_model(lstm_model, test_samples)
    
    print('\nSample predictions:')
    test_texts = ['我喜欢你啊', '你好啊中国', '大笨蛋是你']
    for text in test_texts:
        if len(text) == 5 and '你' in text:
            pos = text.index('你') + 1
            indices = torch.tensor([[char2idx.get(c, 0) for c in text]], dtype=torch.long)
            with torch.no_grad():
                output = lstm_model(indices)
            _, predicted = output.max(1)
            print(f'  Text: {text}, True pos: {pos}, Predicted: {predicted.item()+1}')

main()
