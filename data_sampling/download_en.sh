

python - << 'PY'
import os
os.environ["HF_TOKEN"] = "hf_REDACTED"
# 兼容老接口
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="allenai/c4",
    repo_type="dataset",
    local_dir="/home/work/lyf-sg/projects/vi_zh_data/allenai__c4_en",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=20,
    allow_patterns=[
        "en/**",          # 只要 en
        "README.md",
        "dataset_infos.json",
    ],
)

print("done:", local_dir)
PY





