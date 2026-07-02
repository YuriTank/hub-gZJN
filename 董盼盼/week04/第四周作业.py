
import torch
import math
import numpy as np
import torch.nn as nn
from transformers import BertModel

'''
纯 PyTorch 手动实现 Bert 结构
与官方输出对比 
模型文件下载 https://huggingface.co/models
'''

# 加载官方bert
bert = BertModel.from_pretrained(r"C:\Aistudy\第四周资料\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

# 输入
x = np.array([2450, 15486, 102, 2110])#假想成4个字的句子
torch_x = torch.LongTensor([x])

seqence_output, pooler_output = bert(torch_x)
print(f"bert输出：{seqence_output.shape}, {pooler_output.shape}")
print(f"查看所有的权值矩阵名称:{bert.state_dict().keys()}")  #查看所有的权值矩阵名称
#state_dict 表示模型的所有可学习参数（权重和偏置）的字典。

# Softmax
def softmax(x):
    return torch.softmax(x, dim=-1)

# Gelu 激活函数
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class DiyBert(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = bert.config.num_hidden_layers
        self.load_weights(state_dict)#在对象创建时就完成权重加载

    def load_weights(self, state_dict):
        # Embedding 层 :三种Embedding相加最后再过layerNorm
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"]

        #transformer
        self.transformer_weights = []
        for i in range(self.num_layers):
            q_w = state_dict[f"encoder.layer.{i}.attention.self.query.weight"]
            q_b = state_dict[f"encoder.layer.{i}.attention.self.query.bias"]
            k_w = state_dict[f"encoder.layer.{i}.attention.self.key.weight"]
            k_b = state_dict[f"encoder.layer.{i}.attention.self.key.bias"]
            v_w = state_dict[f"encoder.layer.{i}.attention.self.value.weight"]
            v_b = state_dict[f"encoder.layer.{i}.attention.self.value.bias"]

            attention_output_weight = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"]
            attention_output_bias = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"]
            attention_layer_norm_w = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"]
            attention_layer_norm_b = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"]

            #前馈神经网络有两层线性+GELU激活函数
            intermediate_weight = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"]
            intermediate_bias = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"]
            output_weight = state_dict[f"encoder.layer.{i}.output.dense.weight"]
            output_bias = state_dict[f"encoder.layer.{i}.output.dense.bias"]
            ff_layer_norm_w = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"]
            ff_layer_norm_b = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"]

            self.transformer_weights.append([
                q_w, q_b, k_w, k_b, v_w, v_b,
                attention_output_weight, attention_output_bias,
                attention_layer_norm_w, attention_layer_norm_b,
                intermediate_weight, intermediate_bias,
                output_weight, output_bias,
                ff_layer_norm_w, ff_layer_norm_b
            ])

        # pooler 层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"]

    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x)#词向量
        #torch.arange(len(x)) 创建一个从0开始，长度为len(x)的等差序列张量
        pe = self.get_embedding(self.position_embeddings, torch.arange(len(x)))#位置向量
        #torch.zeros(len(x), dtype=torch.long) 创建一个长度为len(x)的全0张量;
        #在BERT中，0表示第一个句子（单句输入时所有词都是第一个句子）
        te = self.get_embedding(self.token_type_embeddings, torch.zeros(len(x), dtype=torch.long))#句向量

        embedding = we + pe + te
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embedding

    def get_embedding(self, embedding_matrix, x):
        return embedding_matrix[x]

    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        q_w, q_b, k_w, k_b, v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #模型权重初始化预训练模型的权重

        attention_output = self.self_attention(
            x, q_w, q_b, k_w, k_b, v_w, v_b,
            attention_output_weight, attention_output_bias,
            self.num_attention_heads, self.hidden_size
        )

        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)

        feed_forward_x = self.feed_forward(x, intermediate_weight, intermediate_bias, output_weight, output_bias)
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
                       attention_output_weight, attention_output_bias,
                       num_attention_heads, hidden_size):
        q = torch.matmul(x, q_w.T) + q_b#w*x+b 本质就是词向量经过一个全连接层线性处理为Q向量
        k = torch.matmul(x, k_w.T) + k_b#词向量经过一个全连接层线性处理为K向量
        v = torch.matmul(x, v_w.T) + v_b#词向量经过一个全连接层线性处理为v向量

        attention_head_size = hidden_size // num_attention_heads

        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)#4*64
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)#64*4
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)#4*64

        qk = torch.matmul(q, k.transpose(-1, -2))#k.transpose(-1, -2)交换倒数第一和倒数第二位置，其实就是矩阵转置
        qk /= math.sqrt(attention_head_size)
        qk = softmax(qk)

        qkv = torch.matmul(qk, v)
        qkv = qkv.transpose(0, 1).reshape(-1, hidden_size)#多头拼接

        attention = torch.matmul(qkv, attention_output_weight.T) + attention_output_bias#qkv 经过一个全连接层
        return attention

    #多头拆分
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)#[4,12,64]
        x = x.transpose(0, 1)#交换维度顺序 [12,4,64]
        return x

    def feed_forward(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):
        x = torch.matmul(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        x = torch.matmul(x, output_weight.T) + output_bias
        return x

    def layer_norm(self, x, w, b, eps=1e-12):
        mean = x.mean(dim=-1, keepdim=True)#keepdim=True保持维度数量不变 
        std = x.std(dim=-1, keepdim=True, unbiased=False)#nbiased=False是否使用无偏估计 → False 就是：使用总体方差（BERT 官方标准）
        x = (x - mean) / (std + eps)
        x = x * w + b
        return x

    def pooler_output_layer(self, x):
        x = torch.matmul(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = torch.tanh(x)
        return x

    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

# ===================== 运行 =====================
db = DiyBert(state_dict)
db.eval()  # 推理模式

# 转 torch tensor
x_tensor = torch.from_numpy(x).long()

# 自定义 pytorch 版
diy_sequence_output, diy_pooler_output = db(x_tensor)

# 官方版
torch_sequence_output, torch_pooler_output = bert(torch_x)

# 输出对比
print("\n===== 自定义 DiyBert 输出 =====")
print(diy_sequence_output)

print("\n===== 官方 Bert 输出 =====")
print(torch_sequence_output)

# 检查是否一致
print("\n输出是否几乎相同：", torch.allclose(diy_sequence_output, torch_sequence_output[0], atol=1e-2))
