# -*- coding: utf-8 -*-
"""
检查模型权重文件是否完整。

使用方式：
  python src/check_model.py
"""

import os
import sys

from config import MODEL_PATH

# WSL2 路径在 Windows 原生环境不存在时，尝试映射到 Windows 路径
_CHECK_PATH = MODEL_PATH
if not os.path.isdir(_CHECK_PATH) and MODEL_PATH.startswith("/mnt/"):
    drive = MODEL_PATH[5]  # e.g. 'z'
    rest = MODEL_PATH[7:]  # remove '/mnt/z/'
    _CHECK_PATH = f"{drive.upper()}:/" + rest.lstrip("/")

REQUIRED_FILES = {
    "config.json": 100,
    "generation_config.json": 100,
    "merges.txt": 1000,
    "model.safetensors": 500 * 1024 * 1024,  # 约 988 MB
    "tokenizer.json": 1000,
    "tokenizer_config.json": 100,
    "vocab.json": 1000,
}


def main():
    print("检查模型目录:", _CHECK_PATH)
    if not os.path.isdir(_CHECK_PATH):
        print("[错误] 目录不存在:", _CHECK_PATH)
        sys.exit(1)

    ok = True
    for filename, min_size in REQUIRED_FILES.items():
        path = os.path.join(_CHECK_PATH, filename)
        if not os.path.exists(path):
            print("[缺失]", filename)
            ok = False
            continue
        size = os.path.getsize(path)
        size_mb = size / 1024 / 1024
        if size < min_size:
            print("[异常] %s: %.2f MB (小于预期 %.2f MB)" % (filename, size_mb, min_size / 1024 / 1024))
            ok = False
        else:
            print("[正常] %s: %.2f MB" % (filename, size_mb))

    if ok:
        print("\n[OK] 模型文件检查通过")
    else:
        print("\n[FAIL] 模型文件不完整，请运行 python src/download_model.py 重新下载")
        sys.exit(1)


if __name__ == "__main__":
    main()
