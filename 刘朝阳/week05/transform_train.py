# 基于transformer的单向语言模型，完成文本生成

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random

# ============ 数据准备 ============
TRAIN_FILE = 'train_data.txt'

def load_train_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts

TRAIN_TEXT = load_train_data(TRAIN_FILE)

VOCAB = sorted(set(''.join(TRAIN_TEXT)))
VOCAB = ['<pad>', '<bos>', '<eos>'] + VOCAB
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = CHAR2IDX['<pad>']
BOS_IDX = CHAR2IDX['<bos>']
EOS_IDX = CHAR2IDX['<eos>']

# ============ 多头自注意力层（支持因果mask） ============
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        return output

# ============ 前馈神经网络 ============
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ============ Transformer解码器层（单向/因果） ============
class MyTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

# ============ 位置编码 ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============ Transformer单向语言模型 ============
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([
            MyTransformerDecoder(d_model, num_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask.long()
    
    def forward(self, x):
        seq_len = x.size(1)
        mask = self._generate_causal_mask(seq_len, x.device)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        
        for layer in self.decoder_layers:
            x = layer(x, mask)
        
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

def prepare_sequences(texts, max_len=32):
    sequences = []
    for text in texts:
        ids = [BOS_IDX] + [CHAR2IDX[c] for c in text] + [EOS_IDX]
        if len(ids) > max_len:
            ids = ids[:max_len]
        sequences.append(ids)
    return sequences

def make_batches(sequences, batch_size):
    random.shuffle(sequences)
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        max_len = max(len(s) for s in batch)
        padded = [s + [PAD_IDX] * (max_len - len(s)) for s in batch]
        batches.append(padded)
    return batches

# ============ 训练 ============
def train_model(model, texts, epochs=50, batch_size=8, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    sequences = prepare_sequences(texts)
    
    for epoch in range(epochs):
        batches = make_batches(sequences, batch_size)
        total_loss = 0
        
        for batch in batches:
            x = torch.tensor(batch, dtype=torch.long, device=device)
            input_ids = x[:, :-1]
            target_ids = x[:, 1:]
            
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(batches)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model

# ============ 文本生成 ============
def generate_text(model, prompt, max_new_tokens=20, temperature=0.8, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    ids = [BOS_IDX] + [CHAR2IDX[c] for c in prompt if c in CHAR2IDX]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                top_vals = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, top_vals.indices, top_vals.values)
            
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            if next_id.item() == EOS_IDX:
                break
            x = torch.cat([x, next_id], dim=1)
    
    generated_ids = x[0].tolist()
    result = ''.join([IDX2CHAR[i] for i in generated_ids if i not in (BOS_IDX, EOS_IDX, PAD_IDX)])
    return result

def main():
    print(f"词表大小: {VOCAB_SIZE}")
    print(f"训练文本数: {len(TRAIN_TEXT)}")
    
    model = TransformerLM(VOCAB_SIZE, d_model=128, num_heads=4, d_ff=512, n_layers=4, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")
    
    print("\n开始训练...")
    model = train_model(model, TRAIN_TEXT, epochs=50, batch_size=8, lr=3e-4)
    
    print("\n文本生成测试:")
    prompts = ["春天", "夏天", "秋天", "冬天", "小猫", "读书"]
    for prompt in prompts:
        result = generate_text(model, prompt, max_new_tokens=15, temperature=0.8, top_k=5)
        print(f"  输入: '{prompt}' -> 输出: '{result}'")

main()
