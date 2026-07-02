# Week 8 文本匹配作业 —— 使用指南

本文档详细说明如何从零开始复现本作业的全部实验，包括环境准备、数据格式、训练参数含义与常见问题排查。

---

## 1. 环境准备

### 1.1 推荐环境

- Python 3.10+
- PyTorch 2.2.2 + CUDA（已在 NVIDIA RTX 4060 Ti 16GB 验证）
- conda 环境 `ai`

### 1.2 安装依赖

```bash
conda activate ai
pip install torch==2.2.2 transformers==4.39.0 peft==0.11.1 \
    scikit-learn matplotlib tqdm
```

> 若使用其他 PyTorch / transformers 版本，可能需要调整 `dtype` / `torch_dtype` 参数。

### 1.3 本地模型缓存

本作业默认使用 Hugging Face 模型 ID：

- `bert-base-chinese`
- `Qwen/Qwen2-0.5B-Instruct`

首次训练时需要联网下载。如果无法联网，请确保 `~/.cache/huggingface/hub/` 中已存在对应模型缓存，并设置离线环境变量：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Windows（Git Bash）命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_all.py
```

---

## 2. 数据格式

数据集位于 `data/lcqmc/` 与 `data/bq_corpus/`，每个目录下包含三个 JSONL 文件：

```jsonl
{"sentence1": "句子A", "sentence2": "句子B", "label": 1}
{"sentence1": "句子C", "sentence2": "句子D", "label": 0}
```

- `label=1` 表示语义相似
- `label=0` 表示语义不相似

---

## 3. 训练参数说明

### 3.1 BiEncoder

```bash
python src/train_biencoder.py \
    --dataset lcqmc \          # 数据集
    --loss cosine \            # loss 类型：cosine / triplet
    --num_train 10000 \        # 训练样本数（None 表示全量）
    --epochs 2 \               # 训练轮数
    --batch_size 64 \          # 训练 batch size
    --num_hidden_layers 4 \    # 使用 BERT 前 4 层
    --pool mean \              # 池化方式：mean / cls / max
    --max_length 64            # 最大序列长度
```

### 3.2 CrossEncoder

```bash
python src/train_crossencoder.py \
    --dataset lcqmc \
    --num_train 10000 \
    --epochs 2 \
    --batch_size 64 \
    --num_hidden_layers 4 \
    --max_length 128           # CrossEncoder 拼接两句，需要更长长度
```

### 3.3 LLM SFT

```bash
python src_llm/train_sft.py \
    --dataset lcqmc \
    --num_train 2000 \         # 正负各 1000 条平衡采样
    --epochs 1 \
    --batch_size 4 \
    --grad_accum 4 \           # 等效 batch size = 16
    --max_length 128
```

---

## 4. 评估与对比

### 4.1 BERT 方案对比

```bash
python src/compare_methods.py --dataset lcqmc --split validation
```

自动搜索 `outputs/checkpoints/` 下匹配 `{dataset}_{method}_*best.pt` 的权重，按 ntrain 最大优先加载。

### 4.2 SFT 方案评估

```bash
python src_llm/evaluate_sft.py \
    --dataset lcqmc \
    --ckpt_dir outputs/lcqmc_sft_adapter_ntrain2000 \
    --num_samples 500
```

---

## 5. 一键复现

```bash
python run_all.py
```

可用选项：

| 选项 | 说明 |
| --- | --- |
| `--datasets lcqmc bq_corpus` | 指定数据集 |
| `--num_train 10000` | BERT 方法训练样本数 |
| `--num_train_sft 2000` | SFT 训练样本数 |
| `--retrain` | 强制重新训练，覆盖已有 checkpoint |

---

## 6. 常见问题

### 6.1 `OSError: Incorrect path_or_model_id`

原因：Windows 路径中的反斜杠被 transformers 当作转义字符。

解决：本代码已统一使用 Hugging Face 模型 ID 字符串，不再使用本地 Windows 路径。

### 6.2 Hugging Face 下载卡住或超时

原因：网络连接 huggingface.co 不稳定。

解决：

1. 确保模型已缓存到本地。
2. 运行命令前加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

### 6.3 `TypeError: __init__() got an unexpected keyword argument 'dtype'`

原因：当前 transformers 版本使用 `torch_dtype` 而非 `dtype`。

解决：本代码已统一改为 `torch_dtype`。

### 6.4 控制台显示乱码

原因：Windows 默认代码页为 GBK，无法显示中文进度条中的特殊字符。

解决：不影响文件保存与结果；可在脚本开头添加 `chcp 65001` 切换到 UTF-8。

---

## 7. 扩展建议

- 尝试 `--num_train -1` 使用完整训练集，观察 SFT 与 BERT 方案的差距变化。
- 对比 `--pool cls` 与 `--pool mean` 对 BiEncoder 的影响。
- 将 BiEncoder 导出的向量用于 Faiss 检索，体验表示型匹配的可扩展性。
