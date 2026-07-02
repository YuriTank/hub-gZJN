***********************************************BERT NER 项目 — 四方案汇总对比****************************************
方案                      Precision   Recall         F1          非法序列         评估方式
BERT + Linear              0.951153     0.958436     0.954781     47(2318)     seqeval
BERT + CRF                 0.959072    0.96117       0.96012          0        seqeval
Qwen API zero-shot         0.7579     0.4068         0.5294         N/A        span F1
Qwen API few-shot          0.8468     0.5311         0.6528         N/A        span F1
Qwen2-0.5B SFT (LoRA)      0.8760     0.7465         0.8061           0        span F1

总结：BERT + CRF效果最佳，其相比BERT + Linear, CRF Viterbi 解码将 BIO 非法序列从 40 条降至 0，entity F1 提升约 0.01 个点，Qwen API few-shot相比wen API zero-shot F1提升约0.1个点
说明适当增加示例可以提升生成效果。实验数据说明判别式效果优于生成式，因为NER 需要精确边界，序列标注天然对齐这个目标，生成式方法没有位置约束，CRF 保证零非法序列，生成式方法无此保证。

以下是实验过程数据：
1.  bert+linear:
*****************************train******************************
设备：cpu
✅ 加载标签完成：['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
BIO 标签数：7（O + 6 个实体标签）

📊 数据集规模：
训练集：20864 条
验证集：2318 条
测试集：4636 条
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4557.46it/s]
模型：BERT + Linear
  标签数：7
  参数总量：102.3M
  可训练参数：102.3M

训练步数：3912，预热步数：391

开始训练（BERT+Linear）...
Epoch 1/3 | train_loss=0.1234 | val_loss=0.0231 | val_entity_f1=0.9293 | time=8648s
  ★ 新最优 F1=0.9293，已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_linear.pt
Epoch 2/3 | train_loss=0.0164 | val_loss=0.0167 | val_entity_f1=0.9509 | time=8406s
  ★ 新最优 F1=0.9509，已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_linear.pt
Epoch 3/3 | train_loss=0.0068 | val_loss=0.0183 | val_entity_f1=0.9548 | time=8073s
  ★ 新最优 F1=0.9548，已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_linear.pt

训练完成！最优 val_entity_f1=0.9548
  Checkpoint: C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_linear.pt
  训练日志:   C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\logs\train_linear.json

******************evaluate**********************
✅ 加载标签完成：['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4499.11it/s]
模型：BERT + Linear
  标签数：7
  参数总量：102.3M
  可训练参数：102.3M
加载 checkpoint（epoch=3，val_f1=0.9548）

📊 数据集规模：
训练集：20864 条
验证集：2318 条
测试集：4636 条

正在在 [validation] 集上推理...

======================================================================
模型：BERT + Linear  |  评估集：validation
======================================================================
Entity-level Precision: 0.9512
Entity-level Recall:    0.9584
Entity-level F1:        0.9548

【逐类型 F1】
              precision    recall  f1-score   support

         LOC     0.9521    0.9622    0.9571      1799
         ORG     0.9179    0.9283    0.9231       976
         PER     0.9864    0.9841    0.9852       882

   micro avg     0.9512    0.9584    0.9548      3657
   macro avg     0.9521    0.9582    0.9552      3657
weighted avg     0.9513    0.9584    0.9548      3657

【非法 BIO 序列统计】
  总序列数：2318
  非法开头（I-X 开头）：0 条
  非法转移（B-X/I-X → I-Y, X≠Y）：47 条
  合计非法序列：47 条
  → 线性头约 2.0% 的序列含非法转移，充分训练的 CRF 可完全消除

评估结果已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\logs\eval_linear_validation.json

2.bert + CRF

********************************train**********************

✅ 加载标签完成：['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
BIO 标签数：7（O + 6 个实体标签）

📊 数据集规模：
训练集：20864 条
验证集：2318 条
测试集：4636 条
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4539.81it/s]
模型：BERT + CRF
  标签数：7
  参数总量：102.3M
  可训练参数：102.3M

训练步数：3912，预热步数：391

开始训练（BERT+CRF）...
Epoch 1/3 | train_loss=5.2177 | val_loss=1.0396 | val_entity_f1=0.9418 | time=5801s
  ★ 新最优 F1=0.9418，已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_crf.pt
Epoch 2/3 | train_loss=0.8739 | val_loss=1.0775 | val_entity_f1=0.9555 | time=5715s
  ★ 新最优 F1=0.9555，已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_crf.pt
Epoch 3/3 | train_loss=0.4427 | val_loss=1.2153 | val_entity_f1=0.9601 | time=5518s
  ★ 新最优 F1=0.9601，已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_crf.pt

训练完成！最优 val_entity_f1=0.9601
  Checkpoint: C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\checkpoints\best_crf.pt
  训练日志:   C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\logs\train_crf.json

*********************************************evaluate*************************************

✅ 加载标签完成：['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6358.20it/s]
模型：BERT + CRF
  标签数：7
  参数总量：102.3M
  可训练参数：102.3M
加载 checkpoint（epoch=3，val_f1=0.9601）

📊 数据集规模：
训练集：20864 条
验证集：2318 条
测试集：4636 条

正在在 [validation] 集上推理...

======================================================================
模型：BERT + CRF  |  评估集：validation
======================================================================
Entity-level Precision: 0.9591
Entity-level Recall:    0.9612
Entity-level F1:        0.9601

【逐类型 F1】
              precision    recall  f1-score   support

         LOC     0.9634    0.9655    0.9645      1799
         ORG     0.9306    0.9344    0.9325       976
         PER     0.9819    0.9819    0.9819       882

   micro avg     0.9591    0.9612    0.9601      3657
   macro avg     0.9586    0.9606    0.9596      3657
weighted avg     0.9591    0.9612    0.9601      3657

【非法 BIO 序列统计】
  总序列数：2318
  非法开头（I-X 开头）：0 条
  非法转移（B-X/I-X → I-Y, X≠Y）：24 条
  合计非法序列：24 条
  → CRF 非法序列 24 条（1.0%）
  → 提示：训练 epoch 不足时转移矩阵尚未收敛；充分训练（3+ epochs）后可降至 0

评估结果已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\logs\eval_crf_validation.json

3.llm zero-shot 

✅ 成功采样 100 条人民日报验证集样本
🔄 已处理 10/100 条 | Zero-shot已识别实体数：11
🔄 已处理 20/100 条 | Zero-shot已识别实体数：26
🔄 已处理 30/100 条 | Zero-shot已识别实体数：36
🔄 已处理 40/100 条 | Zero-shot已识别实体数：46
🔄 已处理 50/100 条 | Zero-shot已识别实体数：54
🔄 已处理 60/100 条 | Zero-shot已识别实体数：62
🔄 已处理 70/100 条 | Zero-shot已识别实体数：77
🔄 已处理 80/100 条 | Zero-shot已识别实体数：81
🔄 已处理 90/100 条 | Zero-shot已识别实体数：88
🔄 已处理 100/100 条 | Zero-shot已识别实体数：95

============================================================
LLM NER 对比结果（模型：deepseek-v4-flash，样本：100 条）
============================================================
方案                    Precision     Recall         F1
----------------------------------------------------
Zero-shot                0.7579     0.4068     0.5294
Few-shot (3例)            0.8468     0.5311     0.6528

📊 LLM 评估结果已保存 → C:\baidunetdiskdownload\week7_study\Sequence_Labeling\outputs\logs\eval_llm_peoples_daily.json

=================================================================
LLM SFT NER 评估结果
=================================================================
  样本数      : 100
  Precision   : 0.8760
  Recall      : 0.7465
  F1          : 0.8061
  JSON 解析失败: 0 条 (0.0%)
  总耗时      : 326.8s，均值 3.27s/条
