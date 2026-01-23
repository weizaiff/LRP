import os
import json
import random
import heapq
import math
import glob
import subprocess
import tempfile
from multiprocessing import get_context
from tqdm import tqdm
from collections import Counter

# ============== 配置区 ==============
SEED = 42
TARGET = 5000
NPROC = 20

# 你本地 SkyPile 的 jsonl 路径
SKY_GLOB = os.path.expanduser(
    "/home/work/lyf-sg/projects/vi_zh_data/Skywork__SkyPile-150B/data/*.jsonl"
)

OUT = "skypile_zh_uniform_5k.jsonl"

REPORT_EVERY = 50_000

# 可选：长度过滤（建议你开一个区间，避免极短/极长）
MIN_TOKENS = None   # e.g. 150
MAX_TOKENS = None   # e.g. 300
# ===================================


# SkyPile 常见字段候选（你不确定就先用这套）
TEXT_CANDIDATE_KEYS = [
    "text", "content", "raw_content", "document", "doc", "article", "maintext", "body"
]

def _pick_text_field(obj: dict) -> str:
    """Prefer known keys; otherwise pick the longest string field."""
    if not isinstance(obj, dict):
        return None

    for k in TEXT_CANDIDATE_KEYS:
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip():
            return k

    best_k, best_len = None, -1
    for k, v in obj.items():
        if isinstance(v, str):
            l = len(v)
            if l > best_len:
                best_k, best_len = k, l
    return best_k


def _open_stream_cmd(path: str):
    """Return subprocess command to stream lines."""
    # plain jsonl/json
    if path.endswith(".jsonl") or path.endswith(".json"):
        return ["cat", path]
    # compressed (keep for compatibility)
    if path.endswith(".json.gz") or path.endswith(".gz"):
        return ["gzip", "-dc", path]
    if path.endswith(".jsonl.zst") or path.endswith(".zst"):
        return ["zstd", "-q", "-dc", path]
    if path.endswith(".jsonl.xz") or path.endswith(".xz"):
        return ["xz", "-dc", path]
    raise ValueError(f"Unsupported file suffix: {path}")


def _token_len_quick(s: str) -> int:
    """Cheap token length proxy across scripts.
    - If whitespace exists: split
    - Else (often Chinese): approximate by char length / 2
    """
    s = s.strip()
    if not s:
        return 0
    if " " in s or "\t" in s or "\n" in s:
        return len(s.split())
    return max(1, len(s) // 2)


def _worker(rank: int, files: list[str], q, seed: int, target: int,
            min_tokens, max_tokens):
    rnd = random.Random(seed + rank * 1_000_003)

    heap = []
    scanned = 0
    kept = 0
    done_files = 0
    text_key = None

    for fp in files:
        cmd = _open_stream_cmd(fp)
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        try:
            for line in p.stdout:
                scanned += 1
                if scanned % REPORT_EVERY == 0:
                    q.put(("scan", rank, REPORT_EVERY, done_files, len(heap), kept))

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

                # optional length filter
                if min_tokens is not None or max_tokens is not None:
                    tl = _token_len_quick(txt)
                    if min_tokens is not None and tl < min_tokens:
                        continue
                    if max_tokens is not None and tl > max_tokens:
                        continue

                kept += 1

                r = rnd.random()
                if len(heap) < target:
                    heapq.heappush(heap, (r, txt))
                else:
                    if r > heap[0][0]:
                        heapq.heapreplace(heap, (r, txt))
        finally:
            if p.stdout:
                p.stdout.close()
            p.wait()

        done_files += 1
        q.put(("file", rank, 1, done_files, len(heap), kept))

    rem = scanned % REPORT_EVERY
    if rem:
        q.put(("scan", rank, rem, done_files, len(heap), kept))

    # IMPORTANT: 不把 heap 直接塞进 Queue，写临时文件传路径
    tmp = tempfile.NamedTemporaryFile(
        delete=False, prefix=f"uniform_topk_r{rank}_", suffix=".jsonl",
        mode="w", encoding="utf-8"
    )
    tmp_path = tmp.name
    for k, txt in heap:
        tmp.write(json.dumps({"k": k, "text": txt}, ensure_ascii=False) + "\n")
    tmp.close()

    q.put(("done", rank, scanned, done_files, tmp_path, text_key, kept))


def _chunk(lst, n):
    k = math.ceil(len(lst) / n)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def sample_uniform_topk(files_glob: str, out_path: str, nproc: int,
                        seed: int = SEED, target: int = TARGET,
                        min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS):
    files = sorted(glob.glob(files_glob))
    if not files:
        raise RuntimeError(f"No files matched: {files_glob}")

    nproc = min(nproc, len(files))
    groups = _chunk(files, nproc)

    ctx = get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(nproc):
        p = ctx.Process(
            target=_worker,
            args=(rank, groups[rank], q, seed, target, min_tokens, max_tokens),
        )
        p.start()
        procs.append(p)

    pbar = tqdm(desc=f"Scanning ({os.path.basename(out_path)})", unit="lines")
    scanned_total = 0
    files_done_by_worker = [0] * nproc
    heap_sizes = [0] * nproc
    kept_by_worker = [0] * nproc

    done_cnt = 0
    tmp_paths = []
    text_keys = []

    total_files = len(files)

    while done_cnt < nproc:
        msg = q.get()
        tag = msg[0]

        if tag == "scan":
            _, rank, inc, done_files, hsz, kept = msg
            scanned_total += inc
            files_done_by_worker[rank] = done_files
            heap_sizes[rank] = hsz
            kept_by_worker[rank] = kept

            files_done_total = sum(files_done_by_worker)
            pbar.update(inc)
            pbar.set_postfix_str(
                f"files={files_done_total}/{total_files} scanned={scanned_total:,} kept≈{sum(kept_by_worker):,} pool≈{sum(heap_sizes):,}"
            )

        elif tag == "file":
            _, rank, inc_files, done_files, hsz, kept = msg
            files_done_by_worker[rank] = done_files
            heap_sizes[rank] = hsz
            kept_by_worker[rank] = kept
            files_done_total = sum(files_done_by_worker)
            pbar.set_postfix_str(
                f"files={files_done_total}/{total_files} scanned={scanned_total:,} kept≈{sum(kept_by_worker):,} pool≈{sum(heap_sizes):,}"
            )

        elif tag == "done":
            _, rank, scanned, done_files, tmp_path, text_key, kept = msg
            done_cnt += 1
            tmp_paths.append(tmp_path)
            text_keys.append(text_key)
            kept_by_worker[rank] = kept
            pbar.set_postfix_str(
                f"files={sum(files_done_by_worker)}/{total_files} scanned={scanned_total:,} done={done_cnt}/{nproc}"
            )

    pbar.close()

    for p in procs:
        p.join()

    candidates = []
    for tp in tmp_paths:
        with open(tp, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                candidates.append((obj["k"], obj["text"]))
        os.remove(tp)

    candidates.sort(key=lambda x: x[0], reverse=True)
    topk = candidates[:target]

    with open(out_path, "w", encoding="utf-8") as f:
        for _, txt in topk:
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    print(f"[FINAL] wrote {len(topk)} -> {out_path}")
    print(f"[FINAL] scanned_total≈{scanned_total:,} lines, files={total_files}")
    if min_tokens is not None or max_tokens is not None:
        print(f"[FINAL] length_filter: min_tokens={min_tokens} max_tokens={max_tokens}")
    print(f"[FINAL] text_key (mostly): {Counter([k for k in text_keys if k]).most_common(5)}")


if __name__ == "__main__":
    sample_uniform_topk(SKY_GLOB, OUT, NPROC)
