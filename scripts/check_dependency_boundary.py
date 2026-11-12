"""Reject bundled upstream solver sources in the working tree and archives."""
import argparse
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    'src/TSPDroneICP.jl', 'src/main.jl', 'src/TSPDChain.jl', 'src/rings.jl',
    'src/chainlets.jl', 'src/tsp.jl', 'src/utils.jl', 'src/icp.jl', 'src/internals.jl',
    'licenses/TSPDroneICP-original.md', 'licenses/TSPDrone-MIT.txt',
}


def check_paths(names):
    bad = sorted(n for n in names if n in FORBIDDEN or n.startswith('src/vendor/'))
    if bad:
        raise ValueError('Bundled upstream solver files: ' + ', '.join(bad))


def check_boundary(root=ROOT):
    files = [p for p in root.rglob('*') if p.is_file()
             and not any(x in {'.git', '__pycache__', '.venv', 'outputs', '.cache'}
                         for x in p.relative_to(root).parts)]
    check_paths(p.relative_to(root).as_posix() for p in files)
    for path in files:
        if path.suffix == '.jl':
            source = path.read_text(encoding='utf-8-sig')
            if 'using TSPDroneICP' in source or 'module TSPDroneICP' in source:
                raise ValueError(f'Obsolete upstream package implementation/import: {path}')
    project = (root / 'Project.toml').read_text(encoding='utf-8-sig')
    if 'name = "ESNGSDM"' not in project:
        raise ValueError('Expected the project-owned ESNGSDM package')
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    count = check_boundary()
    if args.archive:
        with ZipFile(args.archive) as z:
            if z.testzip() is not None:
                raise ValueError('Archive integrity failure')
            names = [n.removeprefix('ES-NGS-DM/') for n in z.namelist()]
            check_paths(names)
    print(f'DEPENDENCY_BOUNDARY_PASS project_files={count} bundled_solver_sources=0')


if __name__ == '__main__':
    main()
