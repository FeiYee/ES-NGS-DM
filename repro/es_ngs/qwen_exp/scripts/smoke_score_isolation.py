"""P0 scoring isolation smoke.

Same tensors + same ckpt + 4 concurrent scorers + exclusive dirs
⇒ element-wise identical scores and TMP_LEFT=0.

Always:
  - set ESNGS_FORCE_CPU=1 before launching score_server
  - read READY before LOAD
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def write_feats(d: Path, n: int = 8, seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    M, T = 32, 33
    np.save(d / "X.npy", rng.standard_normal((n, M, 24), dtype=np.float32))
    np.save(d / "A.npy", rng.integers(0, 8, size=(n, T, T), dtype=np.int64))
    np.save(d / "action.npy", rng.standard_normal((n, 32), dtype=np.float32))
    np.save(d / "global.npy", rng.standard_normal((n, 16), dtype=np.float32))
    np.save(d / "mask.npy", np.ones((n, T), dtype=np.float32))


def score_loop(score_py: Path, ckpt: Path, feat_dir: Path, out_scores: Path, repeats: int) -> np.ndarray:
    env = os.environ.copy()
    env["ESNGS_FORCE_CPU"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ESNGS_TORCH_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(score_py)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(score_py.parent),
    )
    assert proc.stdin and proc.stdout
    ready = proc.stdout.readline()
    if not ready.startswith("READY"):
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"no READY: {ready!r} stderr={err[:500]}")
    proc.stdin.write(f"LOAD\t{ckpt}\n")
    proc.stdin.flush()
    # may interleave LOADING line
    ok = None
    for _ in range(5):
        line = proc.stdout.readline()
        if line.startswith("OK"):
            ok = line
            break
        if line.startswith("ERR"):
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"LOAD failed: {line.strip()} {err[:500]}")
        if not line:
            break
    if ok is None:
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"LOAD timeout/no OK stderr={err[:500]}")
    last = None
    for _ in range(repeats):
        proc.stdin.write(f"SCORE\t{feat_dir}\t{out_scores}\tnone\n")
        proc.stdin.flush()
        resp = proc.stdout.readline()
        if not resp.startswith("OK"):
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"SCORE failed: {resp.strip()} {err[:500]}")
        last = np.load(out_scores)
        time.sleep(0.002)
    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    assert last is not None
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--score-server", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    args.ckpt = args.ckpt.resolve()
    args.score_server = args.score_server.resolve()
    args.workdir = args.workdir.resolve()

    work = args.workdir / f"isolation_{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    print("WORK", work, flush=True)

    excl_bytes = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for i in range(args.workers):
            feat_dir = work / f"excl_w{i}_{uuid.uuid4().hex}"
            write_feats(feat_dir, seed=123)
            out = feat_dir / "scores.npy"
            futs.append(ex.submit(score_loop, args.score_server, args.ckpt, feat_dir, out, 4))
        for fut in as_completed(futs):
            excl_bytes.append(fut.result())

    ref = excl_bytes[0]
    identical = all(np.array_equal(ref, s) for s in excl_bytes)
    print("EXCLUSIVE_N", len(excl_bytes), "IDENTICAL", identical, "shape", ref.shape, flush=True)

    # cleanup worker dirs; report leftovers
    for p in work.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    left = [p.name for p in work.iterdir()] if work.exists() else []
    # also count nested npy leftovers
    tmp_left = 0
    if work.exists():
        for root, dirs, files in os.walk(work):
            tmp_left += len(files) + len(dirs)
    print("TMP_LEFT", tmp_left, "entries", left, flush=True)

    if identical and len(excl_bytes) == args.workers and tmp_left == 0:
        print("SMOKE_ISOLATION_PASS", flush=True)
        raise SystemExit(0)
    print("SMOKE_ISOLATION_FAIL", flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
