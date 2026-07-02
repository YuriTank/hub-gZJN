"""推理测试脚本"""
import torch
import os
import numpy as np
from multiclass_train import MultiClassClassifier, predict

# 路径：脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载模型
model_path = os.path.join(SCRIPT_DIR, "multiclass_model.pth")
checkpoint = torch.load(model_path, map_location='mps')
model = MultiClassClassifier(
    input_dim=checkpoint['vector_dim'],
    hidden_dim=checkpoint['hidden_dim'],
    num_classes=checkpoint['num_classes']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"模型加载成功: vector_dim={checkpoint['vector_dim']}, hidden_dim={checkpoint['hidden_dim']}")

# 测试推理
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

np.random.seed(123)
test_data = torch.from_numpy(np.random.rand(5, checkpoint['vector_dim']).astype(np.float32))

print("\n测试数据:")
for i, vec in enumerate(test_data):
    true_label = torch.argmax(vec).item()
    pred, probs = predict(model, vec.unsqueeze(0), device)
    print(f"  样本{i}: 向量={vec.numpy().round(3)}, 真标签={true_label}, 预测={pred.item()}, 概率={probs[0].numpy().round(3)}")
