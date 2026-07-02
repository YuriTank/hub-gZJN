# Week 9 作业使用指南

本文档覆盖从环境搭建到每个演示脚本的完整执行流程。

---

## 一、环境要求

| 组件 | 要求 | 说明 |
|------|------|------|
| 系统 | WSL2 Ubuntu 22.04 或原生 Linux | vLLM 不支持 Windows 原生 |
| GPU | NVIDIA 独立显卡 | 显存 ≥ 8GB 即可跑 0.5B 模型 |
| 驱动 | NVIDIA 驱动 566.x 左右 | CUDA 12.x 兼容 |
| Python | 3.10 | 与 vLLM 0.9.2 匹配 |

---

## 二、安装步骤

### 2.1 WSL2 + Ubuntu 22.04

```powershell
# 管理员 PowerShell
wsl --install -d Ubuntu-22.04
```

重启后按提示创建用户名密码。

### 2.2 Ubuntu 内安装基础依赖

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential git curl wget
```

### 2.3 创建虚拟环境并安装 Python 包

```bash
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

cd /mnt/.../week9大模型应用补充知识/homework
pip install -r requirements.txt
```

> 注意：vLLM 0.9.2 + torch 2.7+cu126 是兼容 CUDA 12.x 驱动的推荐组合。若安装最新版 vLLM，可能要求 CUDA 13 / 驱动 580+。

### 2.4 验证环境

```bash
python -c "import vllm, torch; print(vllm.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出包含 `vLLM: 0.9.2`、`CUDA: True` 和 GPU 名称。

---

## 三、配置模型路径

### 方式一：环境变量（推荐）

```bash
export MODEL_PATH="/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"
```

### 方式二：修改 config.py

编辑 `src/config.py`：

```python
DEFAULT_MODEL_PATH = "/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"
```

> 模型路径建议使用 WSL2 内部路径（`/mnt/d/...`），避免跨 9P 文件系统加载模型变慢。

---

## 四、启动 vLLM Server

```bash
bash src/start_server.sh
```

启动过程约 15~30 秒，看到 `Application startup complete` 即成功。

测试服务：

```bash
# 查询模型
curl http://localhost:8000/v1/models

# 简单对话
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-0.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

---

## 五、运行约束解码演示

### 5.1 一键运行全部 demo

```bash
python run_all.py
```

### 5.2 单独运行某个 demo

```bash
python src/demo_guided_choice.py
python src/demo_guided_regex.py
python src/demo_guided_json.py
python src/demo_response_format.py
python src/demo_function_call.py
```

### 5.3 运行吞吐对比

> 需要先停掉 server 释放显存。

```bash
pkill -f 'vllm.entrypoints'
python src/bench_throughput.py
```

运行结束后会生成：
- `outputs/throughput_results.json`
- `outputs/throughput_comparison.png`

---

## 六、常见问题排查

### 6.1 `torch.cuda.is_available()` 返回 False

- 检查是否使用了 vLLM 0.9.2 + torch 2.7+cu126。
- 检查 NVIDIA 驱动是否支持 CUDA 12.x。
- WSL2 中确保 Windows 侧 NVIDIA 驱动已安装。

### 6.2 vLLM 启动时报 CUDA 版本不匹配

```
torch requires CUDA 13.0 but system has 12.x
```

解决：降级到 vLLM 0.9.2 + torch 2.7+cu126。

### 6.3 模型加载路径错误

```
FileNotFoundError: ... Qwen2-0.5B-Instruct/config.json
```

解决：检查 `MODEL_PATH` 环境变量或 `src/config.py` 中的路径是否正确。

### 6.4 demo 运行时连接失败

```
ConnectionRefusedError: [Errno 111] Connection refused
```

解决：先启动 vLLM server。

### 6.5 transformers 5.x 与 vLLM 0.9.2 冲突

若 `requirements.txt` 安装后仍报错 `aimv2` 相关错误，可手动固定：

```bash
pip install transformers==4.52.4
```

