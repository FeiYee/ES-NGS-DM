"""Create an archive from the verified release manifest, excluding run outputs."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from check_dependency_boundary import check_boundary, check_paths

ROOT=Path(__file__).resolve().parents[1]


def main():
    check_boundary()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT.parent / "ES-NGS-DM.zip")
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"choose a new archive path: {args.out}")
    manifest=ROOT / "provenance/SHA256.json"
    entries=json.loads(manifest.read_text(encoding="utf-8"))
    with ZipFile(args.out,"w",compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for relative in sorted(entries):
            path=ROOT / relative
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entries[relative]:
                raise ValueError(f"integrity check failed: {relative}")
            archive.write(path, f"ES-NGS-DM/{relative}")
        archive.write(manifest,"ES-NGS-DM/provenance/SHA256.json")
    with ZipFile(args.out) as archive:
        check_paths(n.removeprefix("ES-NGS-DM/") for n in archive.namelist())
        weights=[x for x in archive.namelist() if Path(x).suffix in {".pt",".pth",".safetensors",".ckpt"}]
        assert weights==["ES-NGS-DM/checkpoints/es_ngs_dm.pt"]
        for i in (1,2,3):
            assert f"ES-NGS-DM/docs/figures/Fig{i}.png" in archive.namelist()
    print(f"RELEASE_ARCHIVE_PASS files={len(entries)+1} bytes={args.out.stat().st_size} path={args.out}")


if __name__=="__main__":
    main()
