
## 多方对比（BQCORPUS validation 集，所有方案均使用 Accuracy + F1，直接可比）
  ┌──────────────────────────────────────────┬──────────┬──────────┐
  │ 方法                                     │ Accuracy │ F1(pos)  │
  ├──────────────────────────────────────────┼──────────┼──────────┤
  │ BiEncoder + CosineEmbeddingLoss          │ 0.8278   │ 0.8275   │
  │ BiEncoder + TripletLoss                  │ **0.8614**│ **0.8614**│
  │ CrossEncoder + CrossEntropyLoss          │ 0.8594   │ 0.8593   │
  │ Qwen API zero-shot                       │ 0.7300   │ 0.4490   │
  │ Qwen2-0.5B SFT（LoRA）                   │ 0.8050   │ 0.8251   │
  └──────────────────────────────────────────┴──────────┴──────────┘
**总结**：从各种方法对比(图中池化方法是mean)可以看出**BiEncoder + TripletLoss**的Accuracy和F1是最优的，这个正和老师文档说的一样数据量大时，Triplet 的优势会更明显优于CosineEmbedding。从对比中也可以看出Qwen API zero-shot的Accuracy和F1是最差的，BiEncoder + TripletLoss的F1比其高了22个点，说明判别式比生成式在文本匹配上仍然占优。

###  BiEncoder CosineEmbeddingLoss  VS  TripletLoss训练
### CosineEmbeddingLoss 的不同池化训练对比
  ┌──────────────────────────────────────────┬──────────┬──────────┐
  │ 方法                                     │ Accuracy │ F1(pos)  │
  ├──────────────────────────────────────────┼──────────┼──────────┤
  │ BiEncoder + CosineEmbeddingLoss (cls)    │ 0.8110   │ 0.8108   │
  │ BiEncoder + CosineEmbeddingLoss（mean）  │ **0.8614**│ **0.8614**│
  │ BiEncoder + CosineEmbeddingLoss（max）   │ 0.7842   │ 0.7840   │
  └──────────────────────────────────────────┴──────────┴──────────┘
**总结**：由于时间原因，仅对比了BiEncoder + CosineEmbeddingLoss的三种池化效果，从表格可以看出mean表现明显优于cls和max，文本匹配这种语义相似度任务需要整体平均看整个句子而不能只能最大的或者只能争个句子整合后的结果，和Sentence-BERT 论文结论：mean pooling 在语义相似度任务优于cls max结论一致。

# CosineEmbeddingLoss训练数据如下：
设备: cpu
Loss 类型: cosine  池化策略: mean  BERT 层数: 12  Epochs: 3

DataLoader 构建中...
  train :  15,000 条,   938 batch
  val   :   5,000 条,   313 batch
  test  :   8,620 条,   539 batch

构建模型...
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3015.21it/s]
模型: BiEncoder (pool=mean, layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)
总训练步数: 2814  Warmup 步数: 281
Epoch 1/3 | train_loss=0.2396 | val_acc=0.7950 val_f1=0.7950 threshold=0.72 | 3323s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\biencoder_cosine_best.pt  (val_f1=0.7950)
Epoch 2/3 | train_loss=0.1763 | val_acc=0.8268 val_f1=0.8268 threshold=0.68 | 3256s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\biencoder_cosine_best.pt  (val_f1=0.8268)
Epoch 3/3 | train_loss=0.1455 | val_acc=0.8272 val_f1=0.8272 threshold=0.70 | 3336s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\biencoder_cosine_best.pt  (val_f1=0.8272)

训练完成。最优 val_f1=0.8272

# TripletLoss训练数据如下:
设备: cpu
Loss 类型: triplet  池化策略: mean  BERT 层数: 12  Epochs: 3

DataLoader 构建中...
  TripletDataset: 构建 34,438 个三元组
  triplet train :  15,000 三元组,   938 batch
  val (pair)    :   5,000 对,       313 batch

构建模型...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 7156.72it/s]
模型: BiEncoder (pool=mean, layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)
总训练步数: 2814  Warmup 步数: 281
Epoch 1/3 | train_loss=0.1061 | val_acc=0.8364 val_f1=0.8364 threshold=0.62 | 5441s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\biencoder_triplet_best.pt  (val_f1=0.8364)
Epoch 2/3 | train_loss=0.0385 | val_acc=0.8600 val_f1=0.8599 threshold=0.52 | 7535s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\biencoder_triplet_best.pt  (val_f1=0.8599)
Epoch 3/3 | train_loss=0.0179 | val_acc=0.8606 val_f1=0.8606 threshold=0.54 | 5130s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\biencoder_triplet_best.pt  (val_f1=0.8606)

训练完成。最优 val_f1=0.8606

###  CrossEncoder训练
设备: cpu
BERT 层数: 12  Epochs: 3  Batch size: 16

DataLoader 构建中...
  train :  15,000 条,   938 batch
  val   :   5,000 条,   313 batch
  test  :   5,000 条,   313 batch

构建模型...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3853.10it/s]
模型: CrossEncoder (layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)
总训练步数: 2814  Warmup 步数: 281
Epoch 1/3 | train_loss=0.5056 train_acc=0.7514 | val_acc=0.8274 val_f1=0.8274 | 4471s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\crossencoder_best.pt  (val_f1=0.8274)
Epoch 2/3 | train_loss=0.3094 train_acc=0.8759 | val_acc=0.8498 val_f1=0.8498 | 16518s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\crossencoder_best.pt  (val_f1=0.8498)
Epoch 3/3 | train_loss=0.1823 train_acc=0.9407 | val_acc=0.8566 val_f1=0.8566 | 3475s
  ✓ 新最优模型已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\checkpoints\crossencoder_best.pt  (val_f1=0.8566)

训练完成。最优 val_f1=0.8566

###  BiEncoder 和CrossEncoder评估结果
设备: cpu
加载 checkpoint: ../outputs/checkpoints/biencoder_cosine_best.pt
训练信息: {'bert_path': 'C:\\baidunetdiskdownload\\week8_study\\progess\\pretrain_models\\bert-base-chinese', 'data_dir': 'C:\\baidunetdiskdownload\\week8_study\\progess\\data\\bq_corpus', 'loss': 'cosine', 'pool': 'mean', 'num_hidden_layers': 12, 'epochs': 3, 'batch_size': 16, 'max_length': 64, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1, 'margin': 0.3}
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2806.42it/s]
模型: BiEncoder (pool=mean, layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

==================================================
BiEncoder 评估结果（test，8620 条）
  最优阈值: 0.65
  Accuracy: 0.8255
  F1      : 0.8253
  AUC     : 0.8964

  图表已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\figures\biencoder_test_sim_dist.png

              precision    recall  f1-score   support

         不相似       0.84      0.80      0.82      4238
          相似       0.81      0.85      0.83      4382

    accuracy                           0.83      8620
   macro avg       0.83      0.83      0.83      8620
weighted avg       0.83      0.83      0.83      8620


设备: cpu
加载 checkpoint: ../outputs/checkpoints/crossencoder_best.pt
训练信息: {'bert_path': 'C:\\baidunetdiskdownload\\week8_study\\progess\\pretrain_models\\bert-base-chinese', 'data_dir': 'C:\\baidunetdiskdownload\\week8_study\\progess\\data\\bq_corpus', 'num_hidden_layers': 12, 'epochs': 3, 'batch_size': 16, 'max_length': 128, 'lr': 2e-05, 'head_lr_mult': 5.0, 'warmup_ratio': 0.1, 'grad_accum': 1}
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4765.85it/s]
模型: CrossEncoder (layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

==================================================
CrossEncoder 评估结果（test，8620 条）
  Accuracy: 0.8476
  F1      : 0.8476

              precision    recall  f1-score   support

         不相似       0.84      0.85      0.85      4238
          相似       0.85      0.84      0.85      4382

    accuracy                           0.85      8620
   macro avg       0.85      0.85      0.85      8620
weighted avg       0.85      0.85      0.85      8620

## 方法对比（三种训练方式）
```bash
cd src
# 运行如下命令
python compare_methods.py
#运行结果如下：
设备: cpu  评估集: validation

=======================================================
加载 biencoder_cosine ...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3793.70it/s]
模型: BiEncoder (pool=mean, layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

=======================================================
加载 biencoder_triplet ...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3555.54it/s]
模型: BiEncoder (pool=mean, layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

=======================================================
加载 crossencoder ...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3235.88it/s]
模型: CrossEncoder (layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

=================================================================
方法                              Accuracy  F1(weighted)    额外信息
-----------------------------------------------------------------
  biencoder_cosine                0.8278        0.8275      threshold=0.61
  biencoder_triplet               0.8614        0.8614      threshold=0.57
  crossencoder                    0.8594        0.8594      argmax

─────────────────────────────────────────────────────────────────
结论速览：
  最高 Accuracy : biencoder_triplet (0.8614)
  最高 F1       : biencoder_triplet  (0.8614)

  Cosine vs Triplet (Δ):
    Accuracy: +0.0336  F1: +0.0339
    → TripletLoss 更优，三元组对语义距离的约束更精确

对比日志 → C:\baidunetdiskdownload\week8_study\progess\outputs\logs\method_comparison.json
  图表已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\figures\method_comparison_bar.png
  图表已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\figures\biencoder_sim_distributions.png
```
## Bad Case 分析与优化方向
>基于 BiEncoder + CosineEmbeddingLoss（threshold=0.51，3 epoch）在 validation 集上的错误分析。
> 总错误 864 条，错误率 10%
| 错误类型 | 数量 | 占总错误 | 说明 |
|---------|-----:|--------:|------|
| FP 假阳性（预测相似，实为不同）| 452 | 52.3% | 模型过度自信"相似" |
| └─ 高置信度错误（Δscore > 0.15）| 317 | 36.7% | 问题最严重，离阈值远 |
| └─ 临界错误（Δscore ≤ 0.15）| 135 | 15.6% | 接近阈值，调整阈值可改善 |
| FN 假阴性（预测不同，实为相似）| 412 | 47.7% | 模型错过真实相似对 |
| └─ 高置信度错误（Δscore > 0.15）| 280 | 32.4% | 需改进模型表示能力 |
| └─ 临界错误（Δscore ≤ 0.15）| 132 | 15.3% | 接近阈值，调阈值可部分改善 |
15.3%
总结：高置信错误：36.7%+32.4% = 69.1%  临界错误：15.6% + 15.3% = 30.9%，大部分都落在了远离阈值的区域，高置信错误需要靠模型/数据来解决。

```bash
cd src
# 分析 BiEncoder（CosineEmbeddingLoss）的错误案例
python analyze_badcases.py
#运行结果如下：
  数据集: validation.jsonl  共 8,620 条
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3713.11it/s]
模型: BiEncoder (pool=mean, layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

整体准确率: 0.8272  错误数: 864

============================================================
Bad Case 汇总  (共 864 个错误)
────────────────────────────────────────────────────────────
  FP 假阳性（预测相似，实际不同）:  452 条
    其中高置信度错误  (Δscore>0.15): 317 条
    其中临界错误     (Δscore≤0.15): 135 条
  FN 假阴性（预测不同，实际相似）:  412 条
    其中高置信度错误  (Δscore>0.15): 280 条
    其中临界错误     (Δscore≤0.15): 132 条

────────────────────────────────────────────────────────────
Bad Case 语言特征分析：

  【FP（假阳性）】共 452 条
    长度差     : 均值=5.8  中位=4
    s1 长度    : 均值=11.9
    s2 长度    : 均值=11.6
    字符 Jaccard: 均值=0.219  （1=完全重叠，0=无共同字符）

  【FN（假阴性）】共 412 条
    长度差     : 均值=5.3  中位=4
    s1 长度    : 均值=11.1
    s2 长度    : 均值=11.9
    字符 Jaccard: 均值=0.202  （1=完全重叠，0=无共同字符）

============================================================

  FP 高置信度错误（score最高的5条） (展示 5 条)：
    score=0.999  | '借款申请一般几天到账'
                  | '一班多少时间放款'

    score=0.998  | '2天都没有收到来电'
                  | '2天没收到电话确认'

    score=0.998  | '什么叫未满足微粒贷的银行审批要求'
                  | '建行'

    score=0.998  | '我有一笔借款正在等银行电话，怎么取消'
                  | '取消借款怎么操作'

    score=0.998  | '没钱还了/流泪'
                  | '你好，系统提示扣款4884怎么扣了4903'


  FP 临界错误（5条） (展示 5 条)：
    score=0.747  | '/睡什么时候有资格'
                  | '合适才能邀请我使用微粒贷'

    score=0.841  | '不可以直接还款吗？'
                  | '你好，我又无法手工还款了'

    score=0.845  | '怎么更换手机号'
                  | '怎么能换新的手机号码呢'

    score=0.802  | '有活动吗'
                  | '就是点上面的抢购链接吗，'

    score=0.786  | '为什么当前没有额度'
                  | '很多朋友都由额度为什么我始终没有额度'


  FN 高置信度错误（score最低的5条） (展示 5 条)：
    score=-0.142  | '多久能审批出结果'
                  | '无法审批'

    score=-0.099  | '为什么我没满足审批要求？'
                  | '不通过怎么办'

    score=-0.088  | '综合评估'
                  | '信用情况'

    score=-0.071  | '怎样什息'
                  | '我今天借款如果明天还进去是不是只算一天的利息？'

    score=-0.067  | '为什么一直借款失败'
                  | '能在后台查出贷款失败的原因吗？'


  FN 临界错误（5条） (展示 5 条)：
    score=0.631  | '银行确定'
                  | '银行电话确认'

    score=0.648  | '无法查到'
                  | '钱包没显示'

    score=0.587  | '借款期限是多久'
                  | '一定是分期还款吗'

    score=0.603  | '不能从零钱里边还钱吗'
                  | '你好，还款不能扣微信余额的吗'

    score=0.662  | '短期(齐鲁稳固21),到期自动续存至下一期，该怎么预约提取？'
                  | '齐鲁稳固21的到期方式？'

  图表已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\figures\biencoder_badcase_dist.png

## 分析 CrossEncoder
python analyze_badcases.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt

# 运行结果如下：
  加载 checkpoint: ..\outputs\checkpoints\crossencoder_best.pt
数据集: validation.jsonl  共 8,620 条
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2917.31it/s]
模型: CrossEncoder (layers=12)
参数量: 102.3M  (BERT 骨干: 102.3M)

整体准确率: 0.8560  错误数: 720

============================================================
Bad Case 汇总  (共 720 个错误)
────────────────────────────────────────────────────────────
  FP 假阳性（预测相似，实际不同）:  374 条
    其中高置信度错误  (Δscore>0.15): 325 条
    其中临界错误     (Δscore≤0.15):  49 条
  FN 假阴性（预测不同，实际相似）:  346 条
    其中高置信度错误  (Δscore>0.15): 308 条
    其中临界错误     (Δscore≤0.15):  38 条

────────────────────────────────────────────────────────────
Bad Case 语言特征分析：

  【FP（假阳性）】共 374 条
    长度差     : 均值=5.6  中位=3
    s1 长度    : 均值=11.6
    s2 长度    : 均值=11.6
    字符 Jaccard: 均值=0.209  （1=完全重叠，0=无共同字符）

  【FN（假阴性）】共 346 条
    长度差     : 均值=5.9  中位=4
    s1 长度    : 均值=11.7
    s2 长度    : 均值=12.1
    字符 Jaccard: 均值=0.211  （1=完全重叠，0=无共同字符）

============================================================

  FP 高置信度错误（score最高的5条） (展示 5 条)：
    score=0.999  | '我忘了密码怎么办'
                  | '密码修改的图发来'

    score=0.999  | '我符合上面条件咋没资格抢'
                  | '什么要求'

    score=0.999  | '开户失败什么意思'
                  | '为什么我开户老是失败'

    score=0.999  | '微粒贷怎么我没'
                  | '和qq微粒贷一样吗'

    score=0.999  | '什么时间开放'
                  | '何时正式开放啊'


  FP 临界错误（5条） (展示 5 条)：
    score=0.573  | '你好我想咨询下，我要更改还款卡号，怎么操作'
                  | '如何变换银行卡还款'

    score=0.543  | '到期没钱还款，怎样算利息'
                  | '今天到期，这个月25号之前还，逾期几天，好吗'

    score=0.549  | '怎么更换手机号'
                  | '怎么能换新的手机号码呢'

    score=0.596  | '还款日期是什么时候'
                  | '什么时候还款呢'

    score=0.546  | '然后获取资格'
                  | '想要资格'


  FN 高置信度错误（score最低的5条） (展示 5 条)：
    score=0.001  | '如果第一次扣不到就算逾期吗？'
                  | '逾期zenme'

    score=0.001  | '微粒贷能停用吗？'
                  | '我想把微粒贷关掉'

    score=0.001  | '你好我要提前全额还款找不到入口'
                  | '如何提前还清？'

    score=0.001  | '我卡里有钱今天是还款曰可发过来都是还款失败是不是时间还不到'
                  | '你好，原来的银行卡换了，我记得前几天提前还了当期的贷款，今天又提醒我还款失败，帮我确认一下！以及如何变更银行卡？'

    score=0.001  | '如何銷卡呢？'
                  | '我想关闭微粒贷，怎么操作？'


  FN 临界错误（5条） (展示 5 条)：
    score=0.455  | '在微信钱包里面吗'
                  | '你们发了好几次，怎就没资格'

    score=0.354  | '没这个功能'
                  | '为何我没有此功能'

    score=0.419  | '换银行卡自动扣款'
                  | '没信用卡可以吗'

    score=0.481  | '为什么重重复复输验证码及支付密码'
                  | '为什么借款密码就是微信支付密码'

    score=0.423  | '为什么综合评估不合格'
                  | '为啥会说我评分不足'

  图表已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\figures\crossencoder_badcase_dist.png

### LLM zero-shot 对比（API 方式）

python llm_compare.py --num_samples 100 --model deepseek-chat

# 运行结果如下：
数据集: validation.jsonl  样本数: 100
  正样本: 33  负样本: 67
模型: deepseek-chat

Prompt 示例：
请判断以下两个问题是否表达相同的意思。只回答"是"或"否"，不要有任何其他内容。

问题1：你好，微粒贷还款可以用微信零钱还吗？
问题2：你好，我微信零钱里面有8000多，想提前还清微粒贷的款，但是点还款那里，没有零钱还款的选项，请问怎么处理？

回答：
──────────────────────────────────────────────────

开始评估（共 100 条，预计 70s）...
  [10/100] 当前准确率（有效预测）: 0.800  解析失败: 0
  [20/100] 当前准确率（有效预测）: 0.750  解析失败: 0
  [30/100] 当前准确率（有效预测）: 0.767  解析失败: 0
  [40/100] 当前准确率（有效预测）: 0.750  解析失败: 0
  [50/100] 当前准确率（有效预测）: 0.780  解析失败: 0
  [60/100] 当前准确率（有效预测）: 0.733  解析失败: 0
  [70/100] 当前准确率（有效预测）: 0.700  解析失败: 0
  [80/100] 当前准确率（有效预测）: 0.688  解析失败: 0
  [90/100] 当前准确率（有效预测）: 0.711  解析失败: 0
  [100/100] 当前准确率（有效预测）: 0.730  解析失败: 0

=======================================================
LLM 评估结果（deepseek-chat，100 条样本）
  准确率 (Accuracy)  : 0.7300
所有指标键： ['accuracy', 'precision_pos', 'recall_pos', 'f1_pos', 'n_valid', 'n_fail']
完整metrics： {'accuracy': 0.73, 'precision_pos': 0.6875, 'recall_pos': 0.3333333333333333, 'f1_pos': 0.4489795918367347, 'n_valid': 100, 'n_fail': 0}
  正例精确率         : 0.6875
  正例召回率         : 0.3333
  正例 F1            : 0.4490
  有效预测数         : 100
  解析失败数         : 0

───────────────────────────────────────────────────────
对比参考（来自 BiEncoder/CrossEncoder 训练结果，val 集全量）：
  指标            | BiEncoder | CrossEncoder | LLM zero-shot
  Accuracy        |  (见训练日志)  |  (见训练日志)  | 0.7300 (100 样本)
  推理速度        |   毫秒级       |   秒级         |  秒级+网络延迟
  可检索（向量）  |    ✓           |    ✗           |   ✗
  需要训练        |    ✓           |    ✓           |   ✗

前 10 条预测错误样本：
  [真:相似 | 预:不相似]
    '为什么我的微粒贷是未通过审核？'  ||  '为什么四次都没通过'
  [真:相似 | 预:不相似]
    '微粒贷开通/色'  ||  '帮我开通微粒贷'
  [真:相似 | 预:不相似]
    '再吗'  ||  '在I不在'
  [真:相似 | 预:不相似]
    '为什么申请了一直没电话审核'  ||  '电话确认需要多久'
  [真:相似 | 预:不相似]
    '借钱后多久银行来电话'  ||  '申请借款后多久能等到确认电话'
  [真:相似 | 预:不相似]
    '这个公众号是干什么的'  ||  '你是什么银行'
  [真:相似 | 预:不相似]
    '还没有接听电话，可以取消借款不'  ||  '没有申请成功可以取消借款吗？'
  [真:相似 | 预:不相似]
    'QQ上有微粒贷，为什么微信上没有微粒贷呢？'  ||  'QQ钱包里的微粒贷和微信里的有区别吗？'
  [真:相似 | 预:不相似]
    '为什么一直不能还款'  ||  '为什么我已还款，但显示失败'
  [真:相似 | 预:不相似]
    '我用其他银行卡还这期的款怎么不可以呢？？为什么要还清才可以用其他银行卡呢？'  ||  '用其它银行卡可还款吗'

结果已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\logs\llm_compare_results.json

## LLM SFT 指令微调（LoRA）

```bash
# ── LoRA 微调──────────────────────────────────────────────
python train_sft.py   
#运行结果如下：
使用设备: cpu  |  微调模式: LoRA 微调
训练集: 5000 条（正2500+负2500，平衡采样）| 验证集（前500条）: 500 条

加载 tokenizer: C:\baidunetdiskdownload\week8_study\progess\pretrain_models\Qwen2-0.5B-Instruct
加载 base model: C:\baidunetdiskdownload\week8_study\progess\pretrain_models\Qwen2-0.5B-Instruct
Loading weights: 100%|█████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 1744.86it/s]
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
总训练步数: 937（batch=4, grad_accum=4, epochs=3, lr=0.0002）

Epoch 1/3 | train_loss=0.1377  val_loss=0.1246 | 4860s
  ✓ 最优LoRA adapter已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\sft_adapter  (val_loss=0.1246)
Epoch 2/3 | train_loss=0.0960  val_loss=0.1172 | 4968s
  ✓ 最优LoRA adapter已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\sft_adapter  (val_loss=0.1172)
Epoch 3/3 | train_loss=0.0684  val_loss=0.1062 | 4968s
  ✓ 最优LoRA adapter已保存 → C:\baidunetdiskdownload\week8_study\progess\outputs\sft_adapter  (val_loss=0.1062)

训练完成。最优 val_loss=0.1062
训练日志 → C:\baidunetdiskdownload\week8_study\progess\outputs\logs\train_sft.json    
## SFT 模型评估

```bash
cd src_llm

python evaluate_sft.py 
#运行结果如下：
LLM SFT 文本匹配评估结果
=================================================================
  样本数      : 200（有效: 200，parse_fail: 0）
  Accuracy    : 0.8050
  F1 (weighted): 0.8051
  F1 (正例)    : 0.8251
  均值耗时     : 0.50s/条（GPU）       
