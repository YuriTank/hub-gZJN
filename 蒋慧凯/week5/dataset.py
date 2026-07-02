import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizer import CharTokenizer
from functools import partial


CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")


class CorpusDataset(Dataset):
    """构建数据集 词的id列表"""

    def __init__(
        self, corpus_path: str, tokenizer: CharTokenizer, max_seq_len: int = 64
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[List[int]] = []

        self._build_dataset_from_corpus(corpus_path)

    def _build_dataset_from_corpus(self, corpus_path: str):
        """从语料库中按行迭代文本 转为id列表"""
        with open(corpus_path, "r", encoding="utf8") as f:
            for line in f:
                cleard_line = line.strip()
                if cleard_line:
                    ids: List[int] = self.tokenizer.encode(cleard_line)
                    if len(ids) > self.max_seq_len:
                        ids = ids[: self.max_seq_len]
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(
    batch: List[List[int]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对齐长度 补充pad 返回一个batch的数据"""
    res: List[torch.Tensor] = []
    mask: List[torch.Tensor] = []
    for ids in batch:
        res.append(torch.tensor(ids, dtype=torch.long))
        mask.append(res[-1] == pad_id)

    batch_dataset = pad_sequence(
        res, batch_first=True, padding_value=pad_id
    )  # batch,seq
    batch_mask = pad_sequence(mask, batch_first=True, padding_value=True)  # batch,seq

    return batch_dataset, batch_mask.long()


if __name__ == "__main__":
    # 1. 初始化 tokenizer
    char_tokenizer = CharTokenizer(corpus_path=CORPUS_PATH)
    # 2. 初始化 dataset
    dataset = CorpusDataset(CORPUS_PATH, char_tokenizer)
    # 3. 初始化 dataloader（batch_size 先设小一点，比如 4）
    dataloader: DataLoader[torch.Tensor] = DataLoader(
        dataset=dataset,
        batch_size=64,
        collate_fn=partial(collate_fn, pad_id=char_tokenizer.pad_id),
    )
    # 4. 取一个 batch 打印
    batch = next(iter(dataloader))
    x = batch[:, :-1]
    y = batch[:, 1:]
    print(f"batch shape: {batch.shape}")
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x[0]: {x[0]}")
    print(f"y[0]: {y[0]}")
    # 验证 y[0][i] == x[0][i+1]
    assert (y[0][:-1] == x[0][1:]).all()
    print("验证通过：y 是 x 左移一位")
