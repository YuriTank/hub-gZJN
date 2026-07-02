# dataset.py 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "peoples_daily"

class PeoplesDailyDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer,
                 label2id: Dict[str, int], max_length: int = 128):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.sentences, self.labels = self._load_data(file_path)

    def _load_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sentences, labels = [], []
        cur_sent, cur_labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if cur_sent:
                        sentences.append(cur_sent)
                        labels.append(cur_labels)
                        cur_sent, cur_labels = [], []
                else:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    char, tag = parts[0], parts[1]
                    cur_sent.append(char)
                    cur_labels.append(tag)
            if cur_sent:
                sentences.append(cur_sent)
                labels.append(cur_labels)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        token_type_ids = encoding.get("token_type_ids", [0] * len(input_ids))
        attention_mask = encoding["attention_mask"]
        word_ids = encoding.word_ids()

        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                if word_idx < len(tags):
                    label_ids.append(self.label2id.get(tags[word_idx], 0))
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx

        label_ids = label_ids[:self.max_length]
        if len(label_ids) < self.max_length:
            label_ids += [-100] * (self.max_length - len(label_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def build_label_schema(data_dir: Path = DATA_DIR):
    """从训练集中自动提取所有标签，确保 O 为 0"""
    train_file = data_dir / "train.txt"
    label_counter = Counter()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 2:
                    label_counter[parts[1]] += 1
    if "O" not in label_counter:
        label_counter["O"] = 0
    # O 放在第一个，其他按出现频率降序
    labels = sorted(label_counter.keys(), key=lambda x: (x != "O", -label_counter[x]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def build_dataloaders(tokenizer, label2id, batch_size=32, max_length=128, data_dir=DATA_DIR):
    train_dataset = PeoplesDailyDataset(data_dir / "train.txt", tokenizer, label2id, max_length)
    val_dataset = PeoplesDailyDataset(data_dir / "dev.txt", tokenizer, label2id, max_length)
    test_dataset = PeoplesDailyDataset(data_dir / "test.txt", tokenizer, label2id, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
