# -*- coding: utf-8 -*-
"""
Week 9 作业一键复现脚本：vLLM 约束解码演示

运行前提：
  1. 已在 WSL2 Ubuntu 中完成环境安装（requirements.txt）
  2. 已启动 vLLM server：bash src/start_server.sh
  3. 已设置 MODEL_PATH 环境变量（或修改 src/config.py 中的默认值）

本脚本会依次运行 5 个约束解码演示脚本：
  - demo_guided_choice.py
  - demo_guided_regex.py
  - demo_guided_json.py
  - demo_response_format.py
  - demo_function_call.py

吞吐对比脚本 bench_throughput.py 需要单独运行（它独占 GPU，且需先停掉 server）。
"""

import subprocess
import sys
import time
import urllib.request

from src.config import API_BASE


def check_server() -> bool:
    """检查 vLLM server 是否已启动"""
    base = API_BASE.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=3) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[警告] 无法连接到 vLLM server: {e}")
        return False


def run_demo(script: str) -> int:
    """运行单个 demo 脚本，返回 exit code"""
    print("\n" + "=" * 70)
    print(f"  运行: {script}")
    print("=" * 70)
    t0 = time.time()
    rc = subprocess.call([sys.executable, f"src/{script}"])
    print(f"  耗时: {time.time() - t0:.1f}s，返回码: {rc}")
    return rc


def main():
    print("Week 9 作业：vLLM 部署与约束解码")
    print(f"API 地址: {API_BASE}")

    if not check_server():
        print("\n[错误] vLLM server 未启动，请先执行：")
        print("  bash src/start_server.sh")
        sys.exit(1)

    demos = [
        "demo_guided_choice.py",
        "demo_guided_regex.py",
        "demo_guided_json.py",
        "demo_response_format.py",
        "demo_function_call.py",
    ]

    results = {}
    for demo in demos:
        results[demo] = run_demo(demo)

    print("\n" + "=" * 70)
    print("  运行汇总")
    print("=" * 70)
    for demo, rc in results.items():
        status = "✅ 成功" if rc == 0 else f"❌ 失败(码{rc})"
        print(f"  {demo:<30} {status}")

    if any(rc != 0 for rc in results.values()):
        sys.exit(1)

    print("\n[提示] 吞吐对比请单独运行：")
    print("  1. 停掉 server：pkill -f 'vllm.entrypoints'")
    print("  2. python src/bench_throughput.py")


if __name__ == "__main__":
    main()
