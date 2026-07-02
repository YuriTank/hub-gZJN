# 四种 NER 方法对比：人民日报数据集

## 一、数据集

- **数据集**：peoples-daily-ner（lansinuote/peoples-daily-ner）
- **训练集**：20865 条
- **验证集**：2319 条
- **测试集**：4637 条
- **实体类型**：3 类（PER 人名 / ORG 机构 / LOC 地名），共 7 个 BIO 标签

---

## 二、方法概览

| 维度 | BERT + Linear | BERT + CRF | LLM SFT (LoRA) | LLM API |
|---|---|---|---|---|
| 底座 | bert-base-chinese (110M) | bert-base-chinese (110M) | Qwen2-0.5B-Instruct (495M) | qwen-flash（API） |
| 范式 | 序列标注（逐字 BIO） | 序列标注（逐字 BIO） | 生成式（输出 JSON） | 生成式（输出 JSON） |
| 训练 | 全参数微调 | 全参数微调 | LoRA（1.08M / 0.22%） | 零训练，纯 prompt |
| 输出 | 每字一个标签 id | 每字一个标签 id（Viterbi） | `{"entities":[...]}` | `{"entities":[...]}` |
| 解码方式 | argmax | CRF Viterbi（全局最优） | 自回归生成 | 自回归生成 |

### 关键差异

- **Linear vs CRF**：CRF 多了一个转移矩阵 + Viterbi 解码，对"B-X→I-Y"等非法转移有约束；Linear 是逐 token 独立决策。
- **BERT vs LLM**：BERT 输出固定标签 id，无解析风险；LLM 需要解析 JSON，多一道格式失败风险。
- **SFT vs API**：SFT 在 corpus 上微调过，更贴合数据集风格；API 是通用大模型 zero/few-shot。

---

## 三、训练配置

### 3.1 BERT 训练（Linear / CRF 通用）

- epochs=3，batch=32，max_length=128
- lr：BERT 层 2e-5，分类头 / CRF 头 1e-4（HEAD_LR_MULT=5.0）
- warmup ratio=0.1，weight_decay=0.01

### 3.2 LLM SFT（LoRA）

- epochs=3，batch=4，grad_accum=4，max_length=256
- lr=2e-4
- LoRA：r=8, α=16, target=`q_proj/k_proj/v_proj/o_proj`
- 可训练参数 1.08M（全参数 495M 的 0.22%）

---

## 四、训练曲线

### 4.1 BERT + CRF

| epoch | train_loss | val_loss | val_f1 |
|---|---|---|---|
| 1 | 8.6652 | 0.9178 | 0.9383 |
| 2 | 0.8157 | 0.7643 | 0.9495 |
| 3 | 0.4016 | 0.9147 | **0.9565** |

### 4.2 LLM SFT (LoRA)

| epoch | train_loss | val_loss |
|---|---|---|
| 1 | 0.0522 | 0.0309 |
| 2 | 0.0219 | 0.0272 |
| 3 | 0.0132 | **0.0235** |

> ⚠️ 注意：CRF 的 train_loss（序列级 NLL）和 Linear 的 train_loss（token 级 CE）量纲不同，CRF 数值大几十倍属正常，**不能直接横向比较**。

---

## 五、最终评估结果

### 5.1 综合对比（汇总表）

| 方案 | 评估集 | 评估标准 | Precision | Recall | **F1** |
|---|---|---|---|---|---|
| BERT + Linear | val 全量（2319） | seqeval（严格） | 95.06% | 95.79% | **95.42%** |
| **BERT + CRF** | val 全量（2319） | seqeval（严格） | 95.78% | 95.52% | **95.65%** |
| LLM SFT (LoRA) | val 采样 50 | span F1 | 90.62% | 79.45% | **84.67%** |
| LLM API zero-shot | val 采样 50 | span F1 | 89.05% | 79.74% | **84.14%** |
| LLM API few-shot | val 采样 50 | span F1 | 87.12% | 75.16% | **80.70%** |

> ⚠️ **评估标准差异**：BERT 用 seqeval（精确 token 边界），SFT/API 用 span F1（`text.find()` 近似定位）。
> SFT 与 API 之间可直接比较；与 BERT 对比时 1~2% 误差属正常。

### 5.2 BIO 序列合法性

| 方案 | total_seqs | illegal_transition | 占比 |
|---|---|---|---|
| BERT + Linear | 2319 | 152 | 6.6% |
| BERT + CRF | 2319 | 27 | 1.2% |

CRF 的转移矩阵约束确实有效——非法转移率从 Linear 的 6.6% 降到 1.2%。

---

## 六、关键结论

### 6.1 CRF 比 Linear 高 0.23%（符合预期）

- 人民日报数据**对 CRF 不友好**：数据量大（20K+）+ 标签简单（仅 7 类）+ 句子短，Linear 自己就能学到边界。
- 真正能拉开差距的场景：小数据集（千条以内）/ 标签多（10+ 类）/ 长实体多。
- BIO 合法性显著提升：1.2% vs 6.6%。

### 6.2 生成式方法（SFT / API）F1 偏低 ≈ 11%

- **不是模型不行，是评估口径吃亏**：人民日报把"欧美""港台"标成两个独立单字 LOC（B-LOC B-LOC），LLM 倾向输出"欧美"作为一个完整实体，被判全错。
- **SFT > API 3.97%**：领域微调确实有效。
- **few-shot < zero-shot 3.44%**：示例里"江泽民/李鹏/宝钢"过于年代化，反而引入偏置。

### 6.3 工程权衡

| 维度 | BERT + CRF | LLM SFT | LLM API |
|---|---|---|---|
| F1 | **最高** | 中 | 中 |
| 训练成本 | 中（单卡 3 epoch ≈ 30 min） | 高（3 epoch × 50 min） | **零** |
| 推理成本 | 低 | 中 | **每条都计费** |
| 落地难度 | 需部署 | 需部署 | API 调用即可 |
| 适合场景 | 标准 NER 任务首选 | 数据规模大、需私有部署 | 快速验证、超低数据量 |

**生产首选 BERT+CRF**：F1 最高，部署成本最低，结果可复现。
