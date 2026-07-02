# Week 8 文本匹配作业

本作业在 **LCQMC** 与 **BQ Corpus** 两个中文语义匹配数据集上，对比四种文本匹配方案：

1. **BiEncoder + CosineEmbeddingLoss**（表示型）
2. **BiEncoder + TripletLoss**（表示型）
3. **CrossEncoder + CrossEntropyLoss**（交互型）
4. **Qwen2-0.5B-Instruct LoRA SFT**（生成式）

实验统一使用 4 层 `bert-base-chinese`（BERT 方案）或 Qwen2-0.5B-Instruct（SFT 方案），控制变量比较不同方法的效果与适用场景。

---

## 目录结构

```
homework/
├── src/                       # BERT 方案代码
│   ├── model.py               # BiEncoder / CrossEncoder
│   ├── dataset.py             # Pair / Triplet / CrossEncoder Dataset
│   ├── train_biencoder.py     # BiEncoder 训练（Cosine / Triplet）
│   ├── train_crossencoder.py  # CrossEncoder 训练
│   ├── evaluate.py            # 评估与相似度分布图
│   └── compare_methods.py     # 多方法对比 + 可视化
├── src_llm/                   # LLM SFT 方案代码
│   ├── train_sft.py           # LoRA SFT 训练
│   └── evaluate_sft.py        # SFT 评估
├── data/                      # LCQMC / BQ Corpus 数据集（本地保留）
├── outputs/                   # 训练输出
│   ├── checkpoints/           # BERT 模型权重（本地保留）
│   ├── figures/               # 对比图表
│   ├── logs/                  # 训练/评估日志
│   ├── *_comparison_table.md  # 对比报告
│   └── *_sft_adapter/         # LoRA adapter（提交小文件）
├── run_all.py                 # 一键复现脚本
├── README.md                  # 项目说明（本文档）
└── user_guide.md              # 详细使用指南
```

---

## 快速开始

```bash
conda activate ai
pip install torch transformers peft scikit-learn matplotlib tqdm

python run_all.py              # 一键复现全部实验（跳过已存在 checkpoint）
python run_all.py --retrain    # 强制重新训练
```

详细参数说明、数据格式、常见问题排查请参考 [user_guide.md](user_guide.md)。

---

## 实验结果

### LCQMC（validation）

| 方法 | Accuracy | F1 (weighted) | F1 (pos) | 说明 |
| --- | --- | --- | --- | --- |
| BiEncoder + Cosine | 0.7217 | 0.7214 | - | threshold=0.92 |
| BiEncoder + Triplet | 0.6984 | 0.6984 | - | threshold=0.91 |
| CrossEncoder | 0.7471 | 0.7428 | - | argmax |
| Qwen2-0.5B SFT (LoRA, 2K) | **0.8400** | **0.8399** | **0.8333** | 生成式 |

### BQ Corpus（validation）

| 方法 | Accuracy | F1 (weighted) | F1 (pos) | 说明 |
| --- | --- | --- | --- | --- |
| BiEncoder + Cosine | 0.7450 | 0.7450 | - | threshold=0.67 |
| BiEncoder + Triplet | 0.7296 | 0.7295 | - | threshold=0.64 |
| CrossEncoder | 0.7585 | 0.7584 | - | argmax |
| Qwen2-0.5B SFT (LoRA, 2K) | **0.7720** | **0.7723** | **0.7841** | 生成式 |

### 关键结论

1. **CrossEncoder > BiEncoder**：在同等 4 层 BERT、2 epoch、10K 训练数据下，CrossEncoder 在两个数据集上均优于 BiEncoder，说明交互式匹配在该任务上更强。
2. **Cosine vs Triplet**：LCQMC 与 BQ Corpus 上 Cosine 均更稳定。小数据 + 短训练周期下，Triplet 难以发挥优势。
3. **SFT 效果显著**：仅 2000 条数据、1 epoch LoRA 微调，Qwen2-0.5B 在 LCQMC 上达到 0.84 Accuracy，远超 BERT 方案；在 BQ Corpus 上也优于 CrossEncoder。
4. **效率对比**：BiEncoder 可向量化检索，推理最快；CrossEncoder 精度高但无法离线建索引；SFT 生成式单条推理约 0.1s，适合精度优先场景。

---

## 输出文件说明

- `outputs/checkpoints/*.pt`：BERT 训练得到的最佳权重（本地保留，不提交）。
- `outputs/figures/*.png`：对比柱状图与 BiEncoder 相似度分布图。
- `outputs/logs/*.json`：训练/评估日志，包含每个 epoch 的 loss、acc、f1 等指标。
- `outputs/*_comparison_table.md`：Markdown 格式对比表。
- `outputs/*_sft_adapter/adapter_config.json`：LoRA 配置（提交）。

> 大文件（`data/`、`*.pt`、SFT adapter 的 safetensors / tokenizer.json / vocab.json / merges.txt）按仓库 `.git/info/exclude` 自动排除，只提交代码、日志、图表与 adapter_config.json 等小文件。

---

## 注意事项

- 脚本文件统一 UTF-8 编码；Windows 控制台可能显示乱码，但不影响文件保存与结果。
- 大模型相关脚本默认离线运行，依赖 Hugging Face 本地缓存。
- 本作业为教学演示，模型与数据规模均受控，请勿直接用于生产环境。
