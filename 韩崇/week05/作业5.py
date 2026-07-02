

import argparse
import math
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_CORPUS = ROOT_DIR / "week4语言模型" / "week4 语言模型" / "循环神经网络语言模型" / "corpus.txt"
DEFAULT_SAVE = Path(__file__).resolve().with_name("transformer_lm.pt")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_corpus(path, max_chars=0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"语料文件不存在: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.strip()
    if max_chars and max_chars > 0:
        text = text[:max_chars]
    if len(text) < 100:
        raise ValueError("语料太短，无法训练语言模型")
    return text


def build_vocab(text):
    chars = sorted(set(text))
    idx2char = [PAD_TOKEN, UNK_TOKEN] + chars
    char2idx = {char: idx for idx, char in enumerate(idx2char)}
    return char2idx, idx2char


def encode_text(text, char2idx):
    unk_id = char2idx[UNK_TOKEN]
    return [char2idx.get(char, unk_id) for char in text]


class CharLanguageModelDataset(Dataset):
    def __init__(self, ids, seq_len):
        if len(ids) <= seq_len:
            raise ValueError("数据长度必须大于 seq_len")
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + 1:index + self.seq_len + 1]
        return x, y


def split_train_val(ids, val_ratio, seq_len):
    split = int(len(ids) * (1 - val_ratio))
    min_part = seq_len + 2
    split = max(min_part, min(split, len(ids) - min_part))
    return ids[:split], ids[split:]


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    @staticmethod
    def build_causal_mask(seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len > self.seq_len:
            raise ValueError(f"输入长度 {seq_len} 不能超过模型最大长度 {self.seq_len}")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        h = self.token_embedding(x) + self.position_embedding(positions)
        mask = self.build_causal_mask(seq_len, x.device)
        h = self.transformer(h, mask=mask)
        h = self.norm(h)
        return self.output(h)


def create_model(args, vocab_size):
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )


def create_optimizer(model, lr):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    except Exception as e:
        raise RuntimeError(
            "创建 AdamW 优化器失败。请先确认使用的是命令行直接运行脚本，"
            "如果在 notebook 或交互环境中运行，请重启内核后再运行。"
            f"原始错误: {type(e).__name__}: {e}"
        ) from e


def run_epoch(model, loader, optimizer, device, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        token_count = y.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 20))


def train(args):
    set_seed(args.seed)
    device = choose_device(args.device)

    text = load_corpus(args.corpus, args.max_chars)
    char2idx, idx2char = build_vocab(text)
    ids = encode_text(text, char2idx)
    train_ids, val_ids = split_train_val(ids, args.val_ratio, args.seq_len)

    train_dataset = CharLanguageModelDataset(train_ids, args.seq_len)
    val_dataset = CharLanguageModelDataset(val_ids, args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = create_model(args, len(idx2char)).to(device)
    optimizer = create_optimizer(model, args.lr)
    best_val_loss = float("inf")

    print(f"device: {device}")
    print(f"语料字符数: {len(text):,}")
    print(f"词表大小: {len(idx2char):,}")
    print(f"训练样本: {len(train_dataset):,}  验证样本: {len(val_dataset):,}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 57)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(model, train_loader, optimizer, device, args.grad_clip)
        with torch.no_grad():
            val_loss, val_ppl = run_epoch(model, val_loader, None, device, args.grad_clip)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *"
            save_checkpoint(args.checkpoint, model, char2idx, idx2char, args)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_ppl:>10.2f}  {val_loss:>10.4f}  {val_ppl:>10.2f}{marker}")

    print(f"训练完成，最佳模型已保存到: {args.checkpoint}")
    return model, char2idx, idx2char, device


def save_checkpoint(path, model, char2idx, idx2char, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "config": {
                "seq_len": args.seq_len,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
            },
        },
        path,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = TransformerLanguageModel(
        vocab_size=len(checkpoint["idx2char"]),
        seq_len=config["seq_len"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint["char2idx"], checkpoint["idx2char"]


def sample_next_token(logits, temperature=0.8, top_k=20):
    logits = logits.clone()
    logits[0] = -float("inf")
    if logits.numel() > 1:
        logits[1] = -float("inf")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k)
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, 1)].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_text(model, char2idx, idx2char, prompt, max_new_chars=80, temperature=0.8, top_k=20, device="cpu"):
    model.eval()
    if not prompt:
        prompt = "中国" if "中" in char2idx else idx2char[2]

    ids = encode_text(prompt, char2idx)
    for _ in range(max_new_chars):
        context = ids[-model.seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1].detach().float().cpu()
        next_id = sample_next_token(logits, temperature=temperature, top_k=top_k)
        ids.append(next_id)

    return "".join(idx2char[token_id] for token_id in ids if idx2char[token_id] not in {PAD_TOKEN, UNK_TOKEN})


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer 单向字符级语言模型")
    parser.add_argument("--mode", choices=["train", "generate", "both"], default="both")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--prompt", default="中国")
    parser.add_argument("--max_new_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=60000, help="训练最多使用多少字符，0 表示使用全部语料")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in {"train", "both"}:
        model, char2idx, idx2char, device = train(args)
    else:
        device = choose_device(args.device)
        model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)

    if args.mode in {"generate", "both"}:
        text = generate_text(
            model=model,
            char2idx=char2idx,
            idx2char=idx2char,
            prompt=args.prompt,
            max_new_chars=args.max_new_chars,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print("\n生成结果:")
        print(text)


if __name__ == "__main__":
    main()

