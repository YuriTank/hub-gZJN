# model.py - 支持线性头和 CRF 层
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF

class BertForTokenClassification(BertPreTrainedModel):
    """BERT + Linear 分类头（与 HuggingFace 原生实现一致）"""
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 只计算非忽略位置的 loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return logits, loss


class BertForTokenClassificationWithCRF(BertPreTrainedModel):
    """BERT + CRF 层"""
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            # CRF 需要忽略 -100 位置，但 CRF 层不支持直接 mask；我们通过 attention_mask 让 CRF 忽略 padding
            # 注意：labels 中 -100 的位置也需要被排除，但 CRF 要求标签形状完整。我们简单将 -100 替换为 0（不会影响 loss 计算？）
            # 更好的方法：将 -100 位置的标签临时改为 0，并设置 mask 为 False（但其实 attention_mask 已经 mask 了 padding）
            # 这里简化：CRF 使用 attention_mask 作为有效位置标志，标签中的 -100 位置实际上不会被 CRF 使用因为 attention_mask=0？
            # 稳妥起见，我们手动构建一个 mask：有效位置 = attention_mask=1 且 labels!=-100
            mask = (labels != -100) & (attention_mask == 1)
            # CRF 要求输入形状 (batch, seq_len)，且标签必须完整，但我们可以将 -100 替换为 0，在 loss 计算时通过 mask 忽略
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0  # 临时填充，但会被 mask 忽略
            loss = -self.crf.forward(emissions, safe_labels, mask=mask, reduction='mean')
        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids):
        """返回预测的标签 id 序列（list of list）"""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            emissions = self.classifier(sequence_output)
            # 使用 attention_mask 作为有效位置掩码
            pred_ids = self.crf.decode(emissions, mask=attention_mask == 1)
        return pred_ids


def build_model(use_crf: bool, bert_path: str, num_labels: int, dropout: float = 0.1):
    """根据 use_crf 构建相应模型"""
    from transformers import BertConfig
    config = BertConfig.from_pretrained(bert_path)
    config.hidden_dropout_prob = dropout
    config.num_labels = num_labels

    if use_crf:
        model = BertForTokenClassificationWithCRF.from_pretrained(
            bert_path, config=config, num_labels=num_labels
        )
    else:
        model = BertForTokenClassification.from_pretrained(
            bert_path, config=config, num_labels=num_labels
        )
    return model
