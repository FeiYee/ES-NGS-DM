"""Verify file hashes, fixed evaluation lists, and instance-grouped data splits."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from check_dependency_boundary import check_boundary

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    check_boundary()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-manifest", action="store_true")
    args = p.parse_args()
    manifest = ROOT / "provenance/SHA256.json"
    if args.write_manifest:
        files = [x for x in ROOT.rglob("*") if x.is_file() and
                 not any(part in {"__pycache__", ".git", "outputs", ".venv"} for part in x.relative_to(ROOT).parts)
                 and x != manifest]
        files = sorted(files)
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(digest, files))
        payload = {x.relative_to(ROOT).as_posix(): value for x, value in zip(files, values)}
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    hashes = json.loads(manifest.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(digest, (ROOT / relative for relative in hashes)))
    for (relative, expected), got in zip(hashes.items(), actual):
        assert got == expected, f"hash mismatch: {relative}"
    index = rows(ROOT / "data/training/state_index_all.csv")
    assert len(index) == 480
    instances = {r["instance_id"] for r in index}
    assert len(instances) == 40
    seen = set()
    for fold in range(1, 6):
        items = rows(ROOT / f"data/splits/supplement/fold{fold}.csv")
        train = {r["instance_id"] for r in items if r["split"] == "train"}
        val = {r["instance_id"] for r in items if r["split"] == "val"}
        assert not train & val and train | val == instances
        assert not seen & val
        seen |= val
    assert seen == instances
    weight_files = [p for p in ROOT.rglob("*") if p.is_file()
                    and p.suffix in {".pt", ".pth", ".safetensors", ".ckpt"}
                    and "outputs" not in p.relative_to(ROOT).parts]
    assert weight_files == [ROOT / "checkpoints/es_ngs_dm.pt"], "expected only the final Synthetic checkpoint"
    evaluation = rows(ROOT / "data/splits/paper_evaluation.csv")
    assert sum(r["benchmark"] == "synthetic" for r in evaluation) == 30
    assert sum(r["benchmark"] == "lade" for r in evaluation) == 8
    for name in ("synthetic", "lade"):
        geometry = rows(ROOT / f"data/{name}/{'manifest_all.csv' if name == 'synthetic' else 'manifest.csv'}")
        lookup = {r["case_id"]: r for r in geometry}
        for r in evaluation:
            if r["benchmark"] == name:
                row = lookup[r["case_id"]]
                assert digest(ROOT / f"data/{name}" / row["geometry_file"]) == row["geometry_sha256"]
    print(f"RELEASE_INTEGRITY_PASS files={len(hashes)} states=480 grouped_folds=5 evaluation_cases=38")


if __name__ == "__main__":
    main()
