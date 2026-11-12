# External solver dependencies

The source distributed here contains the ES-NGS-DM decision pipeline and its
energy and synchronization extensions. The initialization routines are external
dependencies, downloaded directly from their owners rather than copied into
this repository.

## Install

```bash
python -m pip install -r requirements.txt
python scripts/setup_solver_dependencies.py
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Git is required for the setup command. It creates pinned checkouts of
[TSPDroneICP.jl](https://github.com/kaist-comet/TSPDroneICP.jl) and
[TSPDrone.jl](https://github.com/kaist-comet/TSPDrone.jl) under
`~/.es_ngs_dm/upstream/`. The command checks every loaded source file against
`configs/upstream_dependencies.toml` and refuses to overwrite modified checkouts.
The original licenses remain with those external checkouts.

To use another location, set `ESNGS_UPSTREAM_ROOT` for both setup and Julia.

PowerShell:

```powershell
$env:ESNGS_UPSTREAM_ROOT = 'D:\ExternalDependencies\ESNGSDM'
python scripts/setup_solver_dependencies.py
julia --project=. -e "using ESNGSDM"
```

Linux or macOS:

```bash
export ESNGS_UPSTREAM_ROOT=/path/outside/the/repository/esngsdm
python scripts/setup_solver_dependencies.py
julia --project=. -e 'using ESNGSDM'
```

Both setup and the Julia loader reject dependency checkouts inside the project.
Release packaging includes neither those checkouts nor solver binaries.

## Runtime boundary

`src/ESNGSDM.jl` loads the project-owned extensions. Its integration adapter
reads the pinned external initializer and partitioning files without copying
their implementations. The adapter omits imports for unused optional Concorde
and LKH entry points. The published search explicitly selects FI initialization;
the upstream NICP model is not loaded. Network downloads occur only when the
setup command is explicitly run, never during model scoring or search.

The Python scorer, student training, Qwen preference pipeline and saved-pool
ranking do not require these Julia solver checkouts.
