#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN=""
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"   # 兼容老接口

python - << 'PY'
import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    raise RuntimeError("HF_TOKEN is empty. Please export HF_TOKEN before running.")
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

local_dir = "/home/work/lyf-sg/projects/vi_zh_data/allenai__c4_vi"

snapshot_download(
    repo_id="allenai/c4",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=10,
    allow_patterns=[
        "multilingual/c4-vi.*.json.gz",            # 训练 shards (vi)
        "multilingual/c4-vi-validation*.json.gz",  # 验证集（如果有）
        "README.md",
        "dataset_infos.json",
    ],
)

print("done:", local_dir)
PY
