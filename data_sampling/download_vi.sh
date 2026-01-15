python - << 'PY'
import os
os.environ["HF_TOKEN"] = "hf_REDACTED"
# 兼容老接口
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
from huggingface_hub import snapshot_download
repo_id = "Symato/c4_vi-filtered_200GB"
local_dir = "/home/work/lyf-sg/projects/vi_zh_data/Symato__c4_vi-filtered_200GB"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 真拷贝，便于迁移
    token=True,                    # 用已登录token / 环境变量
    resume_download=True,
    max_workers=20,                # 视网络/磁盘调 8~32
)
print("done:", local_dir)
PY
