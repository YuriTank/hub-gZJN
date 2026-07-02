# Week5: 基于 Transformer Decoder 的自回归语言模型（对话机器人）

## 任务目标
训练一个 Decoder-only 的类 GPT 模型，基于语料库 `corpus.txt` 进行 Next Token Prediction 预训练，最终能够进行对话生成。

## 关键设计决策
- **架构**：Decoder-only（GPT 风格），而非 BERT。原因：Next Token Prediction + 对话生成需要单向（因果）注意力。
- **Tokenizer**：字级别（char-level），词表大小约 4500。理由：语料仅 4MB，字级别实现简单、OOV 少。
- **特殊 Token**：`<pad>`(0), `<unk>`(1), `<eos>`(2)，强制排在词表最前面，保证与 PyTorch API 的兼容性。

## 已完成
### Step 1: Tokenizer (`tokenizer.py`)
- 字级别词表构建
- 编解码一致性验证通过
- 特殊 token 固定 id（pad=0, unk=1, eos=2）
- 已加入 `pad_id` / `eos_id` / `unk_id` 属性，避免硬编码

### Step 2: Dataset + DataLoader (`dataset.py`)
- 整行作为样本，超长截断、不足 pad
- Next Token Prediction 标签构造（x=ids[:-1], y=ids[1:]）
- 自定义 `collate_fn`（`functools.partial` 传 `pad_id`）
- batch 验证通过：x/y 形状对齐，y 为 x 左移一位

### Step 3: 模型改造（`model.py`）—— 进行中
基于 week4 `diy_transformer.py` 复制改造，需完成：
- [ ] 修复 `nn.ModuleList` 直接调用的 bug
- [ ] Attention 中加入 Causal Mask（下三角掩码）
- [ ] 添加 LM Head（`nn.Linear(d_model, vocab_size)`）
- [ ] 添加 `generate()` 自回归生成方法

## 待完成
### Step 4: 训练循环
- CrossEntropyLoss（ignore_index=pad_id）
- 训练循环 + 验证 loss 下降
- 模型保存

### Step 5: 对话生成
- 加载训练好的权重
- 自回归解码 + 停止条件（`<eos>` 或最大长度）

## 运行方式
```bash
cd 蒋慧凯/week5
python tokenizer.py    # 测试 tokenizer
python dataset.py      # 测试 Dataset + DataLoader
python model.py        # 测试模型（待完成）
```

## 备注
- 对模型质量和语料规模要求不高，侧重掌握完整流程
- 工程规范优先：固定特殊 token id、避免魔法数字、严格类型注解和注释
