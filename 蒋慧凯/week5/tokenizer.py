import os
from typing import Set, List, Dict


class CharTokenizer:
    """
    基于字级别的最小化 Tokenizer，支持编解码和特殊 Token 处理
    """

    token_pad = "<pad>"
    token_unk = "<unk>"
    token_eos = "<eos>"
    special_chars = [token_pad, token_unk, token_eos]  # pad放在第一位,便于后续流程

    def __init__(self, corpus_path: str):
        # 读语料，构建词表
        self.corpus_path = corpus_path
        self.vocab = []
        self.idx2char: Dict[int, str] = {}
        self.char2idx: Dict[str, int] = {}
        self.max_seq_len = -1
        # 自动构建词表
        self.build_vocab()

    def build_vocab(self):
        # 词表
        chars_set: Set[str] = set()
        with open(self.corpus_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                self.max_seq_len = max(self.max_seq_len, len(line))
                chars_set = chars_set.union(set(line))
        chars_set = chars_set - set(self.special_chars)  # 去掉可能会重复的特殊字符
        self.vocab = self.special_chars + sorted(
            list(chars_set)
        )  # 确保特殊字符排在最前面
        # 映射
        for idx, c in enumerate(self.vocab):
            self.idx2char[idx] = c
            self.char2idx[c] = idx
        # 特殊字符id
        self.pad_id = self.char2idx[self.token_pad]
        self.unk_id = self.char2idx[self.token_unk]
        self.eos_id = self.char2idx[self.token_eos]

    def encode(self, text: str) -> List[int]:
        """将文本编码为 id 列表，自动在末尾追加 <eos>"""
        res: List[int] = []
        for c in text:
            token_id = self.char2idx.get(c, self.unk_id)
            res.append(token_id)
        res.append(self.eos_id)
        return res

    def decode(self, ids: List[int]) -> str:
        """将 id 列表解码为文本，跳过 <pad>，遇到 <eos> 停止"""
        res: List[str] = []
        for token_id in ids:
            token = self.idx2char.get(token_id, self.token_unk)
            if token == self.token_pad:
                continue
            elif token != self.token_eos:
                res.append(token)
            else:
                break
        return "".join(res)

    @property
    def vocab_size(self):
        # 返回 len(self.vocab)
        return len(self.vocab)


if __name__ == "__main__":
    # 测试代码
    tokenizer = CharTokenizer(os.path.join(os.path.dirname(__file__), "corpus.txt"))
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"前10个词表内容: {tokenizer.vocab[:10]}")

    text = "在收盘前1小时"
    ids = tokenizer.encode(text)
    print(f"编码结果: {ids}")
    print(f"最后一个id是否为1（<eos>）: {ids[-1] == tokenizer.eos_id}")

    converted_text = tokenizer.decode(ids)
    print(f"解码结果: {converted_text}")
    print(f"编解码是否一致: {converted_text == text}")
