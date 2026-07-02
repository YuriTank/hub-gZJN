
# 2. 模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel
from pathlib import Path


# 
def _load_bert(bert_path: str) -> BertModel:
    """加载预训练的 BERT 模型，关闭日志。"""
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    # 直接加载预训练模型，保持原有权重和配置
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert

# 模型定义
class BertNER(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        # hidden_size = self.bert.config.hidden_size = 768
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor, # 输入的 token ids，形状 (B, L)
        attention_mask: torch.Tensor, # 输入的 attention mask，形状 (B, L)
        token_type_ids: torch.Tensor, # 输入的 token type ids，形状 (B, L)
        labels: torch.Tensor = None, # 可选的标签，形状 (B, L)，用于计算损失
    ) -> tuple[torch.Tensor, torch.Tensor | None]: # 返回 (logits, loss)，其中 logits 形状 (B, L, num_labels)，loss 是标量
        outputs = self.bert( # 调用 BERT 模型，返回 (last_hidden_state, pooler_output, hidden_states, attentions)
            input_ids=input_ids, # 输入的 token ids，形状 (B, L)
            attention_mask=attention_mask, # 输入的 attention mask，形状 (B, L)
            token_type_ids=token_type_ids, # 输入的 token type ids，形状 (B, L)
            return_dict=True, # 返回字典形式的结果
        )
        seq_output = outputs.last_hidden_state  # (B, L, H)
        logits = self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy( # 计算交叉熵损失，适用于 token 分类任务
                logits.view(-1, self.num_labels), # 将 logits 从 (B, L, num_labels) 展平为 (B*L, num_labels)
                labels.view(-1), # 将 labels 从 (B, L) 展平为 (B*L)
                ignore_index=-100,# 在计算损失时忽略标签为 -100 的位置（特殊 token 和非首子词）
            )
        return logits, loss


class BertCRFNER(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF
        # 加载 BERT 模型 + CRF 层
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()  # CRF 要求 BoolTensor

        loss = None
        if labels is not None:
            # CRF 不支持 ignore_index，将 -100 替换为 0（PAD 位置被 mask 屏蔽，不影响梯度）
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            # crf() 返回对数似然（正值），取负得到损失
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        return emissions, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi 解码，返回 list[list[int]]，每条序列长度等于实际token数（不含PAD）。"""
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


def build_model(
    use_crf: bool,
    bert_path: str,
    num_labels: int,
    dropout: float = 0.1,
) -> nn.Module:
    """模型工厂函数。"""
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model
