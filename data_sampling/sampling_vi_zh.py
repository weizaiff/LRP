import os
import json
import random
import heapq
import math
import glob
import subprocess
from multiprocessing import get_context
from tqdm import tqdm

# ============== 配置区 ==============
SEED = 42
TARGET = 5000
NPROC = 20

# 输入：改成你的真实目录
ZH_GLOB = os.path.expanduser("~/lyf-sg/projects/vi_zh_data/shjwudp__chinese-c4/data/*.jsonl.zst")
VI_GLOB = os.path.expanduser("~/lyf-sg/projects/vi_zh_data/Symato__c4_vi-filtered_200GB/*.jsonl.xz")

# 输出文件名
OUT_ZH = "chinese_c4_uniform_5k.jsonl"
OUT_VI = "c4_vi_uniform_5k.jsonl"

# 进度上报频率（每扫描多少行上报一次）
REPORT_EVERY = 50_000
# ===================================


def _pick_text_field(obj: dict) -> str:
    """Pick longest string field as text field."""
    best_k, best_len = None, -1
    for k, v in obj.items():
        if isinstance(v, str):
            l = len(v)
            if l > best_len:
                best_k, best_len = k, l
    return best_k


def _open_stream_cmd(path: str):
    """Return subprocess command to stream decompressed lines."""
    if path.endswith(".jsonl.zst"):
        # -q quiet, -c stdout
        return ["zstd", "-q", "-dc", path]
    if path.endswith(".jsonl.xz"):
        # -d decompress, -c stdout
        return ["xz", "-dc", path]
    raise ValueError(f"Unsupported file suffix: {path}")


def _worker(rank: int, files: list[str], q, seed: int, target: int):
    rnd = random.Random(seed + rank * 1_000_003)

    # min-heap of (key, text)
    heap = []
    scanned = 0
    done_files = 0
    text_key = None

    for fp in files:
        cmd = _open_stream_cmd(fp)
        # text mode with utf-8; ignore decode errors to avoid crash on rare bad bytes
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        try:
            for line in p.stdout:
                scanned += 1
                if scanned % REPORT_EVERY == 0:
                    q.put(("scan", rank, REPORT_EVERY, done_files, len(heap)))

                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if text_key is None and isinstance(obj, dict):
                    text_key = _pick_text_field(obj)

                if not text_key or not isinstance(obj, dict):
                    continue

                txt = obj.get(text_key, "")
                if not isinstance(txt, str) or not txt:
                    continue

                key = rnd.random()

                if len(heap) < target:
                    heapq.heappush(heap, (key, txt))
                else:
                    if key > heap[0][0]:
                        heapq.heapreplace(heap, (key, txt))
        finally:
            # ensure process exits
            if p.stdout:
                p.stdout.close()
            p.wait()

        done_files += 1
        q.put(("file", rank, 1, done_files, len(heap)))  # file-level update

    # flush remaining scanned count not reported (optional but nice)
    rem = scanned % REPORT_EVERY
    if rem:
        q.put(("scan", rank, rem, done_files, len(heap)))

    q.put(("done", rank, scanned, done_files, heap, text_key))


def _chunk(lst, n):
    k = math.ceil(len(lst) / n)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def sample_uniform_topk(files_glob: str, out_path: str, nproc: int):
    files = sorted(glob.glob(files_glob))
    if not files:
        raise RuntimeError(f"No files matched: {files_glob}")

    # If file count < nproc, cap workers
    nproc = min(nproc, len(files))

    groups = _chunk(files, nproc)

    ctx = get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(nproc):
        p = ctx.Process(target=_worker, args=(rank, groups[rank], q, SEED, TARGET))
        p.start()
        procs.append(p)

    pbar = tqdm(desc=f"Scanning ({os.path.basename(out_path)})", unit="lines")
    scanned_total = 0
    files_done_total = 0

    heap_sizes = [0] * nproc
    files_done_by_worker = [0] * nproc

    done_cnt = 0
    all_heaps = []
    text_keys = []

    total_files = len(files)

    while done_cnt < nproc:
        msg = q.get()
        tag = msg[0]

        if tag == "scan":
            _, rank, inc, done_files, hsz = msg
            scanned_total += inc
            files_done_by_worker[rank] = done_files
            heap_sizes[rank] = hsz
            files_done_total = sum(files_done_by_worker)

            pbar.update(inc)
            pbar.set_postfix_str(
                f"files={files_done_total}/{total_files}  scanned={scanned_total:,}  pool≈{sum(heap_sizes):,}"
            )

        elif tag == "file":
            _, rank, inc_files, done_files, hsz = msg
            files_done_by_worker[rank] = done_files
            heap_sizes[rank] = hsz
            files_done_total = sum(files_done_by_worker)
            pbar.set_postfix_str(
                f"files={files_done_total}/{total_files}  scanned={scanned_total:,}  pool≈{sum(heap_sizes):,}"
            )

        elif tag == "done":
            _, rank, scanned, done_files, heap_list, text_key = msg
            done_cnt += 1
            all_heaps.append(heap_list)
            text_keys.append(text_key)

            # 轻微修正 postfix
            pbar.set_postfix_str(
                f"files={files_done_total}/{total_files}  scanned={scanned_total:,}  done={done_cnt}/{nproc}"
            )

    pbar.close()

    for p in procs:
        p.join()

    # merge all candidates: <= TARGET * nproc
    candidates = []
    for h in all_heaps:
        candidates.extend(h)

    candidates.sort(key=lambda x: x[0], reverse=True)
    topk = candidates[:TARGET]

    # write final
    with open(out_path, "w", encoding="utf-8") as f:
        for _, txt in topk:
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    print(f"[FINAL] wrote {len(topk)} -> {out_path}")
    print(f"[FINAL] scanned_total≈{scanned_total:,} lines, files={total_files}")


if __name__ == "__main__":
    # 中文
    #sample_uniform_topk(ZH_GLOB, OUT_ZH, NPROC)
    # 越南
    sample_uniform_topk(VI_GLOB, OUT_VI, NPROC)
