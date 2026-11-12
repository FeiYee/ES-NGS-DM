# Third-party notices

## External solver dependencies

[TSPDroneICP.jl](https://github.com/kaist-comet/TSPDroneICP.jl), by Jae Hyeok Lee,
Minjun Kim, Jinkyoo Park, Changhyun Kwon and contributors, and
[TSPDrone.jl](https://github.com/kaist-comet/TSPDrone.jl) supply the external
FI/ICP initializer and partitioning routines. Their original MIT licenses
remain in their upstream checkouts. Neither project's source is redistributed
in this repository or its release archives.

`scripts/setup_solver_dependencies.py` obtains pinned checkouts directly from
their owners, outside this project. Revisions and source hashes are recorded
in `configs/upstream_dependencies.toml`. `src/upstream_backend.jl` is integration
code that loads those separately installed files. This project's Julia package
is named `ESNGSDM`; it is not an upstream TSPDroneICP distribution.

`TravelingSalesmanHeuristics` and `Distances` are also external dependencies.
Their licenses apply. The FI initialization does not install Concorde, LKH or
PyCall. Historical environment snapshots are metadata, not bundled packages.
No solver binaries or upstream neural models are redistributed.

## Qwen

The optional offline decision model uses
[Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), revision
`8faed761d45a263340a0528343f099c05c9a4323`, under
[Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE).
The base model is downloaded separately. Qwen fine-tuning and decision-head
source code is included; no Qwen model or adapter weights are bundled.
The offline backbone is truncated to eight layers for decision-head training.
The online delivery solver uses the lightweight ESNGSCritic, not the Qwen backbone.

## LaDe

The eight demand-grounded scenarios derive from the Hangzhou delivery data in
[Cainiao-AI/LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe), whose dataset
card specifies Apache-2.0. Credit: LaDe dataset authors / Cainiao-AI.
See [the official project](https://github.com/wenhaomin/LaDe) and
the [dataset paper](https://arxiv.org/abs/2306.10675).

The released scenario geometries contain centered local AOI coordinates, not
raw parcel records, GPS trajectories, or personal addresses. Station positions,
depot placement, battery parameters and drone operations are simulation
assumptions added by this project. Scenario identifiers preserve the original
selection keys for reproducibility. The upstream license continues to apply
to derived demand data. The complete Apache-2.0 text is included in
`licenses/Qwen-Apache-2.0.txt`; this is the same license text, not an assertion
that Qwen owns LaDe data.

## Changes in the public distribution

The release adds relative-path entry points, fixed protocol configuration,
fail-fast neural scoring, space-safe scorer communication, data verification,
and usage documentation. Absolute machine paths have been removed from public
metadata. Deployment tensor values are unchanged; their provenance is recorded
in `provenance/deployment_checkpoint.json`.
