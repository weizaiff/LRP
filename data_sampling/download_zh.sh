python - << 'PY'
import os
os.environ["HF_TOKEN"] = "hf_REDACTED"
# 兼容老接口
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

from huggingface_hub import snapshot_download
repo_id = "shjwudp/chinese-c4"
local_dir = "/home/work/lyf-sg/projects/vi_zh_data/shjwudp__chinese-c4"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=True,                    # 即使不 gated 也无妨
    resume_download=True,
    max_workers=20,
)
print("done:", local_dir)
PY
