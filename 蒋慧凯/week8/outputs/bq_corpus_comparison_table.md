## BQ_CORPUS 三种方法对比

| 方法 | Accuracy | F1 (weighted) | Threshold | AUC-ROC |
| --- | --- | --- | --- | --- |
| biencoder_cosine | 0.7450 | 0.7450 | 0.67 | 0.8229 |
| biencoder_triplet | 0.7296 | 0.7295 | 0.64 | 0.8017 |
| crossencoder | 0.7585 | 0.7584 | - | - |

- 柱状图: `figures\bq_corpus_acc_f1.png`
- 相似度分布图: `figures\bq_corpus_sim_distribution.png`
