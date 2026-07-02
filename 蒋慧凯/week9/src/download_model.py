# -*- coding: utf-8 -*-
"""
下载 Qwen2-0.5B-Instruct 模型权重到本地。

使用 curl 下载（支持断点续传），默认从 hf-mirror.com（国内镜像）下载。

使用方式：
  python src/download_model.py --output ./pretrain_models/Qwen2-0.5B-Instruct
"""

import argparse
import os
import subprocess
import sys

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

REPOS = [
    "https://hf-mirror.com/Qwen/Qwen2-0.5B-Instruct/resolve/main",
    "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main",
]


def download_with_curl(url: str, dest: str, max_time: int = 1800) -> bool:
    """使用 curl 下载，支持断点续传"""
    print(f"  下载: {url}")
    print(f"  保存: {dest}")
    cmd = [
        "curl", "-C", "-", "-L", "--max-time", str(max_time),
        "-o", dest, url,
    ]
    rc = subprocess.call(cmd)
    return rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="./pretrain_models/Qwen2-0.5B-Instruct",
        help="模型保存目录",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"目标目录: {output_dir}\n")

    for filename in MODEL_FILES:
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[跳过] {filename} 已存在\n")
            continue

        ok = False
        for repo in REPOS:
            url = f"{repo}/{filename}"
            if download_with_curl(url, dest):
                ok = True
                break
            print(f"  镜像 {repo} 下载失败，尝试下一个...")

        if not ok:
            print(f"[错误] 无法下载 {filename}")
            sys.exit(1)
        print()

    print("全部下载完成。")


if __name__ == "__main__":
    main()
