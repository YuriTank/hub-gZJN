import torch
from transformers import BertModel

from DivBert import DivBert

if __name__=='__main__':

    model = BertModel.from_pretrained('./bert-base-chinese')
    state = model.state_dict()
    embedding_word = state['embeddings.word_embeddings.weight']
    embedding_position = state['embeddings.position_embeddings.weight']
    embedding_token_type = state['embeddings.token_type_embeddings.weight']
    embedding_layerNorm_weight = state['embeddings.LayerNorm.weight']
    embedding_layerNorm_bias = state['embeddings.LayerNorm.bias']

    hidden_layer_num = model.config.num_hidden_layers
    layer = []
    for i in range(hidden_layer_num):
        layer.append({
            'qw': state[f'encoder.layer.{i}.attention.self.query.weight'],
            'qb': state[f'encoder.layer.{i}.attention.self.query.bias'],
            'kw': state[f'encoder.layer.{i}.attention.self.key.weight'],
            'kb': state[f'encoder.layer.{i}.attention.self.key.bias'],
            'vw': state[f'encoder.layer.{i}.attention.self.value.weight'],
            'vb': state[f'encoder.layer.{i}.attention.self.value.bias'],
            'w1': state[f'encoder.layer.{i}.attention.output.dense.weight'],
            'b1': state[f'encoder.layer.{i}.attention.output.dense.bias'],
            'norm1w': state[f'encoder.layer.{i}.attention.output.LayerNorm.weight'],
            'norm1b': state[f'encoder.layer.{i}.attention.output.LayerNorm.bias'],
            'feed_w1': state[f'encoder.layer.{i}.intermediate.dense.weight'],
            'feed_b1': state[f'encoder.layer.{i}.intermediate.dense.bias'],
            'feed_w2': state[f'encoder.layer.{i}.output.dense.weight'],
            'feed_b2': state[f'encoder.layer.{i}.output.dense.bias'],
            'norm2w': state[f'encoder.layer.{i}.output.LayerNorm.weight'],
            'norm2b': state[f'encoder.layer.{i}.output.LayerNorm.bias'],
        })
    pooler_dense_weight = state['pooler.dense.weight']
    pooler_dense_bias = state['pooler.dense.bias']

    bert = DivBert( word_embeddings=embedding_word,
        position_embeddings=embedding_position,
        token_type_embeddings=embedding_token_type,
        embedding_layer_norm_weight=embedding_layerNorm_weight,
        embedding_layer_norm_bias=embedding_layerNorm_bias,
        encoder_layers=layer,
        pooler_weight=pooler_dense_weight,
        pooler_bias=pooler_dense_bias,
        hidden_size=model.config.hidden_size,        # 768
        num_attention_heads=model.config.num_attention_heads)


    model.eval()
    input_ids = torch.tensor([[1,2,3,4]])  # 需要 batch 维度
    x_, cls_ = model(input_ids)
    x, cls = bert.forward(input_ids)

    print(f'结果是否相似：{torch.std(x_-x).item() < 0.00001}')




