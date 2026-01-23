#!/usr/bin/env bash
set -euo pipefail

# ====== Token 配置 ======
export HF_TOKEN=""
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"   # 兼容老接口

python - << 'PY'
import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    raise RuntimeError("HF_TOKEN is empty. Please export HF_TOKEN before running.")
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

local_dir = "/home/work/lyf-sg/projects/vi_zh_data/Skywork__SkyPile-150B"

snapshot_download(
    repo_id="Skywork/SkyPile-150B",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=10,

    # SkyPile 是大文件 shard，按格式抓是最稳的
    allow_patterns=[
        "*.parquet",
        "*.jsonl",
        "*.jsonl.zst",
        "README.md",
        "dataset_infos.json",
    ],
)

print("done:", local_dir)
PY
