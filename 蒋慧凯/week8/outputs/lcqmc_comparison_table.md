## LCQMC 三种方法对比

| 方法 | Accuracy | F1 (weighted) | Threshold | AUC-ROC |
| --- | --- | --- | --- | --- |
| biencoder_cosine | 0.7217 | 0.7214 | 0.92 | 0.8034 |
| biencoder_triplet | 0.6984 | 0.6984 | 0.91 | 0.7805 |
| crossencoder | 0.7471 | 0.7428 | - | - |

- 柱状图: `figures\lcqmc_acc_f1.png`
- 相似度分布图: `figures\lcqmc_sim_distribution.png`
