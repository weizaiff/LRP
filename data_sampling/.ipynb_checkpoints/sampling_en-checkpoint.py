import os
import json
import random
import heapq
import math
import glob
import gzip
from multiprocessing import get_context
from tqdm import tqdm

# ============ 配置 ============
DATA_DIR = os.path.expanduser("~/lyf-sg/projects/vi_zh_data/allenai__c4_en/en")
FILE_GLOB = os.path.join(DATA_DIR, "c4-train.*.json.gz")

SEED = 42
TARGET = 5000
NPROC = 20

REPORT_EVERY = 100_000   # 每扫描多少行更新一次进度
OUT = "c4_en_uniform_5k.jsonl"
# ==============================


def pick_text_field(obj: dict) -> str:
    """C4-en 固定有 text，但这里留保险"""
    if "text" in obj:
        return "text"
    best_k, best_len = None, -1
    for k, v in obj.items():
        if isinstance(v, str):
            if len(v) > best_len:
                best_k, best_len = k, len(v)
    return best_k


def worker(rank: int, files: list[str], q):
    rnd = random.Random(SEED + rank * 1_000_003)

    heap = []  # min-heap (key, text)
    scanned = 0
    done_files = 0
    text_key = None

    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                scanned += 1
                if scanned % REPORT_EVERY == 0:
                    q.put(("scan", rank, REPORT_EVERY, done_files, len(heap)))

                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if text_key is None:
                    text_key = pick_text_field(obj)
                    if not text_key:
                        continue

                txt = obj.get(text_key, "")
                if not isinstance(txt, str) or not txt:
                    continue

                key = rnd.random()

                if len(heap) < TARGET:
                    heapq.heappush(heap, (key, txt))
                else:
                    if key > heap[0][0]:
                        heapq.heapreplace(heap, (key, txt))

        done_files += 1
        q.put(("file", rank, 1, done_files, len(heap)))

    # flush remainder
    rem = scanned % REPORT_EVERY
    if rem:
        q.put(("scan", rank, rem, done_files, len(heap)))

    q.put(("done", rank, scanned, done_files, heap))


def chunk(lst, n):
    k = math.ceil(len(lst) / n)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def main():
    files = sorted(glob.glob(FILE_GLOB))
    assert len(files) == 1024, f"expect 1024 shards, got {len(files)}"

    nproc = min(NPROC, len(files))
    groups = chunk(files, nproc)

    ctx = get_context("spawn")
    q = ctx.Queue()
    procs = []

    for rank in range(nproc):
        p = ctx.Process(target=worker, args=(rank, groups[rank], q))
        p.start()
        procs.append(p)

    pbar = tqdm(desc="C4-en full scan (uniform sampling)", unit="lines")
    scanned_total = 0
    files_done_by_worker = [0] * nproc
    heap_sizes = [0] * nproc

    done_cnt = 0
    all_heaps = []

    total_files = len(files)

    while done_cnt < nproc:
        msg = q.get()
        tag = msg[0]

        if tag == "scan":
            _, rank, inc, done_files, hsz = msg
            scanned_total += inc
            files_done_by_worker[rank] = done_files
            heap_sizes[rank] = hsz
            pbar.update(inc)
            pbar.set_postfix_str(
                f"files={sum(files_done_by_worker)}/{total_files}  "
                f"scanned={scanned_total:,}  pool≈{sum(heap_sizes):,}"
            )

        elif tag == "file":
            _, rank, _, done_files, hsz = msg
            files_done_by_worker[rank] = done_files
            heap_sizes[rank] = hsz
            pbar.set_postfix_str(
                f"files={sum(files_done_by_worker)}/{total_files}  "
                f"scanned={scanned_total:,}  pool≈{sum(heap_sizes):,}"
            )

        elif tag == "done":
            _, rank, scanned, done_files, heap = msg
            done_cnt += 1
            all_heaps.append(heap)
            pbar.set_postfix_str(
                f"files={sum(files_done_by_worker)}/{total_files}  "
                f"scanned={scanned_total:,}  done={done_cnt}/{nproc}"
            )

    pbar.close()

    for p in procs:
        p.join()

    # merge top-k
    candidates = []
    for h in all_heaps:
        candidates.extend(h)

    candidates.sort(key=lambda x: x[0], reverse=True)
    topk = candidates[:TARGET]

    with open(OUT, "w", encoding="utf-8") as f:
        for _, txt in topk:
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    print(f"[FINAL] wrote {len(topk)} -> {OUT}")
    print(f"[FINAL] scanned_total≈{scanned_total:,} lines")


if __name__ == "__main__":
    main()
