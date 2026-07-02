import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
import requests
import os
import zipfile
import gc
import time

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# ==================== 1. 获取中等规模新闻语料（约10,000条） ====================
def get_medium_news_corpus(max_docs=10000):
    """
    优先从 THUCNews 下载前 max_docs 条新闻
    如失败则使用内置扩充语料（约3000条合成新闻）
    """
    if os.path.exists("THUCNews"):
        # 已有 THUCNews 文件夹，直接读取
        texts = []
        for root, dirs, files in os.walk("THUCNews"):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) > 50 and not line.startswith("http"):
                                texts.append(line)
                            if len(texts) >= max_docs:
                                break
                    if len(texts) >= max_docs:
                        break
            if len(texts) >= max_docs:
                break
        if len(texts) >= 1000:
            print(f"从 THUCNews 加载了 {len(texts)} 条新闻")
            return texts[:max_docs]
    
    # 如果 THUCNews 不存在或不足，尝试下载
    if not os.path.exists("THUCNews"):
        print("正在下载 THUCNews 子集（约 5MB）...")
        try:
            # 使用清华开源镜像（轻量级）
            url = "https://thuctc.thunlp.org/static/THUCNews.zip"
            r = requests.get(url, stream=True, timeout=30)
            with open("THUCNews.zip", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile("THUCNews.zip", "r") as zip_ref:
                zip_ref.extractall("THUCNews")
            os.remove("THUCNews.zip")
            print("下载解压完成")
            # 递归读取
            texts = []
            for root, dirs, files in os.walk("THUCNews"):
                for file in files:
                    if file.endswith(".txt"):
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if len(line) > 50:
                                    texts.append(line)
                                if len(texts) >= max_docs:
                                    break
                        if len(texts) >= max_docs:
                            break
                if len(texts) >= max_docs:
                    break
            print(f"下载后加载了 {len(texts)} 条新闻")
            return texts[:max_docs]
        except Exception as e:
            print(f"下载失败：{e}，使用内置合成语料")
    
    # 最终 fallback：基于15条新闻扩展至 3000 条（质量尚可）
    base_news = [
        "近日国家发展改革委发布最新数据显示一季度国内生产总值同比增长百分之五点三国民经济开局良好",
        "人工智能领域取得重大突破科学家成功开发出可自我进化的机器学习算法应用于自动驾驶系统",
        "全球气候变化峰会于本周在日内瓦召开多国代表达成碳减排新协议承诺2030年前实现可再生能源占比百分之五十",
        "中国空间站迎来新一批宇航员将开展为期六个月的太空科学实验包括微重力材料制备和生物培育",
        "新能源汽车销量持续攀升比亚迪推出新一代固态电池技术续航里程突破一千公里",
        "央行宣布下调存款准备金率零点二五个百分点释放长期资金约五千亿元支持实体经济发展",
        "教育部发布新版义务教育课程标准强化人工智能和编程教育从2026年秋季学期开始实施",
        "华为发布鸿蒙操作系统最新版本实现多设备无缝协同全球装机量突破八亿",
        "乒乓球世锦赛中国队包揽全部五项冠军年轻选手展现出强大竞技实力",
        "故宫博物院推出线上数字展厅利用虚拟现实技术让观众足不出户欣赏珍贵文物",
        "长三角地区开通首条超级高铁试验线时速达到一千公里上海至杭州仅需十五分钟",
        "中俄东线天然气管道累计输气量突破两千亿立方米有效保障冬季能源供应",
        "第六届中国国际进口博览会在上海开幕来自一百二十五个国家的三千家企业参展",
        "科研团队在南海发现新型深海珊瑚礁生态系统中含有多种未知海洋生物",
        "全国多地启动数字人民币试点应用场景覆盖交通医疗教育等民生领域",
    ]
    # 数据增强生成 3000 条
    synonyms = {
        "发布": ["公布", "发表"], "突破": ["进展", "跨越"], "开展": ["进行", "推进"],
        "推出": ["发布", "上线"], "表示": ["称", "指出"]
    }
    expanded = []
    for _ in range(200):
        for news in base_news:
            chars = list(news)
            for i, ch in enumerate(chars):
                if ch in synonyms and random.random() < 0.2:
                    chars[i] = random.choice(synonyms[ch])
            if random.random() < 0.3 and len(news) > 20:
                pos = random.randint(10, len(news)-10)
                chars.insert(pos, "，")
            expanded.append("".join(chars))
    random.shuffle(expanded)
    print(f"使用内置合成语料，共 {len(expanded)} 条新闻（质量中等）")
    return expanded[:max_docs]

news_texts = get_medium_news_corpus(max_docs=10000)
print(f"实际使用新闻条数: {len(news_texts)}")

# 拼接用于训练的长文本
TRAIN_TEXT = "。".join(news_texts) + "。"

# ==================== 2. 字符级分词器（vocab=3000，覆盖常用汉字） ====================
class CharTokenizer:
    def __init__(self, text, vocab_size=3000):
        counter = Counter(text)
        most_common = counter.most_common(vocab_size - 4)
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token_to_idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.idx_to_token = {i: tok for i, tok in enumerate(self.special_tokens)}
        for ch, _ in most_common:
            if ch not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[ch] = idx
                self.idx_to_token[idx] = ch
        self.vocab_size = len(self.token_to_idx)
        print(f"词汇表大小: {self.vocab_size}")
    
    def encode(self, text):
        return [self.token_to_idx.get(ch, self.token_to_idx['<UNK>']) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_token.get(idx, '<UNK>') for idx in indices])

tokenizer = CharTokenizer(TRAIN_TEXT, vocab_size=3000)

# ==================== 3. 构建训练序列（序列长度128，步长64，增加样本） ====================
def build_sequences(text, tokenizer, seq_len=128, stride=64):
    tokens = tokenizer.encode(text)
    sequences = []
    for i in range(0, max(1, len(tokens) - seq_len), stride):
        inp = tokens[i:i+seq_len]
        tgt = tokens[i+1:i+seq_len+1]
        if len(inp) == seq_len and len(tgt) == seq_len:
            sequences.append((inp, tgt))
    # 确保至少有一些样本
    if len(sequences) < 100:
        for i in range(0, len(tokens) - seq_len, seq_len):
            inp = tokens[i:i+seq_len]
            tgt = tokens[i+1:i+seq_len+1]
            if len(inp) == seq_len and len(tgt) == seq_len:
                sequences.append((inp, tgt))
    print(f"生成 {len(sequences)} 个训练样本")
    return sequences

sequences = build_sequences(TRAIN_TEXT, tokenizer, seq_len=128, stride=64)
split = int(0.95 * len(sequences))
train_seqs = sequences[:split]
val_seqs = sequences[split:]

class TextDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx][0], dtype=torch.long), torch.tensor(self.seqs[idx][1], dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TextDataset(train_seqs), batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(TextDataset(val_seqs), batch_size=batch_size, shuffle=False)

# ==================== 4. 高效 Transformer 模型（参数量约 800 万） ====================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # 因果掩码（Causal Mask）
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        y = self.proj(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class EfficientNewsLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, nhead=6, num_layers=4, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 初始化
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if repetition_penalty != 1.0:
                for i in range(idx.size(1)):
                    token = idx[0, i].item()
                    logits[0, token] /= repetition_penalty
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = EfficientNewsLM(tokenizer.vocab_size, d_model=192, nhead=6, num_layers=4)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")

# ==================== 5. 快速训练（混合精度 + 余弦退火，15个epoch ≈ 15-20分钟 GPU） ====================
def train_20min(model, train_loader, val_loader, epochs=15, lr=3e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    )
    
    # 混合精度（仅 GPU）
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = total_loss / len(train_loader)
        ppl = math.exp(min(avg_loss, 10))
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}: train loss = {avg_loss:.4f}, perplexity = {ppl:.2f}, time = {elapsed:.1f}s')
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        val_avg = val_loss / len(val_loader)
        val_ppl = math.exp(min(val_avg, 10))
        print(f'Validation loss = {val_avg:.4f}, PPL = {val_ppl:.2f}')
        
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), 'best_news_model_20min.pt')
            print(f'  -> 保存最佳模型')
        
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    model.load_state_dict(torch.load('best_news_model_20min.pt', map_location=device))
    print("训练完成，已加载最佳模型。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device == 'cpu':
    print("警告：CPU 训练可能超过 20 分钟，建议使用 GPU")
train_20min(model, train_loader, val_loader, epochs=15, lr=3e-4, device=device)

# ==================== 6. 生成测试 ====================
def generate_news(prompt, max_new_tokens=100, temperature=0.75, top_p=0.9):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.token_to_idx['<UNK>']]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_new_tokens, temperature, top_p, repetition_penalty=1.1)
    return tokenizer.decode(output_ids[0].cpu().tolist())

test_prompts = ["人工智能", "国家发改委", "新能源汽车", "中国空间站"]
for p in test_prompts:
    gen = generate_news(p, max_new_tokens=100, temperature=0.75)
    print("\n" + "="*60)
    print(f"提示词: {p}")
    print(f"生成文本: {gen}")
    print("="*60)
