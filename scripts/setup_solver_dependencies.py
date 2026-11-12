"""Install pinned upstream solver sources outside this repository."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def git(folder, *args):
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    proc = subprocess.run(['git', '-C', str(folder), *args], env=env,
                          text=True, encoding='utf-8', capture_output=True)
    if proc.returncode:
        raise RuntimeError(f'git {args[0]} failed: {proc.stderr.strip()}')
    return proc.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get(
        'ESNGS_UPSTREAM_ROOT', Path.home() / '.es_ngs_dm' / 'upstream')))
    args = parser.parse_args()
    external = args.root.expanduser().resolve()
    if external == ROOT or external.is_relative_to(ROOT):
        raise ValueError('Dependencies must be installed outside the repository.')
    config = tomllib.loads((ROOT / 'configs/upstream_dependencies.toml').read_text(encoding='utf-8'))
    external.mkdir(parents=True, exist_ok=True)
    for name, spec in config['dependencies'].items():
        target = external / spec['directory']
        if target.exists():
            if not (target / '.git').is_dir():
                raise ValueError(f'Refusing to overwrite a non-checkout directory: {target}')
            if git(target, 'rev-parse', 'HEAD') != spec['revision']:
                raise ValueError(f'Existing dependency has a different revision: {target}')
            if git(target, 'status', '--porcelain'):
                raise ValueError(f'Existing dependency has local edits: {target}')
        else:
            target.mkdir()
            git(target, 'init')
            git(target, 'config', 'core.autocrlf', 'false')
            git(target, 'remote', 'add', 'origin', spec['url'])
            git(target, 'fetch', '--depth', '1', 'origin', spec['revision'])
            git(target, 'checkout', '--detach', 'FETCH_HEAD')
        for relative, expected in spec['files'].items():
            path = target / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Upstream source integrity check failed: {path}')
        print(f'EXTERNAL_DEPENDENCY_PASS {name} revision={spec["revision"]}')
    print(f'ESNGS_UPSTREAM_ROOT={external}')
    print('Dependencies are installed separately and are not included in ES-NGS-DM archives.')


if __name__ == '__main__':
    main()
