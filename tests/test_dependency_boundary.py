"""Packaging and dependency-setup boundary tests; no network downloads."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_dependency_boundary import check_boundary, check_paths


class DependencyBoundaryTests(unittest.TestCase):
    def test_public_tree_has_no_upstream_solver_sources(self):
        self.assertGreater(check_boundary(), 0)

    def test_vendored_and_foundation_files_are_rejected(self):
        for path in ['src/vendor/TSPDrone/tsp_ep_all.jl', 'src/icp.jl', 'src/TSPDroneICP.jl']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_paths([path])

    def test_project_extensions_are_allowed(self):
        check_paths(['src/ESNGSDM.jl', 'src/energy_icp.jl', 'repro/es_ngs/model.py'])

    def test_setup_refuses_repository_as_dependency_root(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/setup_solver_dependencies.py'),
                                 '--root', str(ROOT)], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('outside the repository', result.stderr)


if __name__ == '__main__':
    unittest.main()
