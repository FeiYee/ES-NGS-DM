"""Download the pinned official Qwen base model; no API key is required."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "models/Qwen2.5-1.5B")
    args = p.parse_args()
    config = json.loads((ROOT / "configs/qwen_decision.json").read_text(encoding="utf-8"))
    snapshot_download(repo_id=config["repo"], revision=config["revision"],
                      local_dir=args.out, allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"])
    print(f"QWEN_BASE={args.out.resolve()}")


if __name__ == "__main__":
    main()
