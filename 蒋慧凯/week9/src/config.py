"""
统一配置：模型路径、服务地址等。

使用方式：
  1. 设置环境变量 MODEL_PATH（推荐）
     export MODEL_PATH="/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"
  2. 不设置则使用下方的默认路径，请按实际情况修改
"""

import os

# 模型本地路径（WSL2 内建议使用 /mnt/... 路径，避免跨 9P 文件系统）
# 默认指向本项目同级目录 pretrain_models/，与课程仓库结构保持一致
DEFAULT_MODEL_PATH = "/mnt/z/LearningDocs/八斗AI/pretrain_models/Qwen2-0.5B-Instruct"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# OpenAI 兼容 API 地址
API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "qwen2-0.5b")
