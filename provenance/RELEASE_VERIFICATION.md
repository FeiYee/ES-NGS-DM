# Release verification, 2026-11-12

| Check | Result |
|---|---|
| Distributed learned weights | Exactly one: `checkpoints/es_ngs_dm.pt` |
| Weight origin | Historical final Synthetic-trained paper deployment model |
| Tensor preservation | All 68 state-dict tensors bitwise unchanged, 205,253 parameters |
| Architecture and feature encoder | SHA256 matches the recorded experiment audit |
| Frozen geometries | 900 Synthetic + 8 LaDe files deserialize and match their geometry hashes |
| Training data | 480 supplementary states, 40 instances, 28,800 labeled candidates |
| Grouped splits | Five fixed instance-level train/validation folds |
| Paper run records | 150 accepted Synthetic runs and 40 accepted LaDe runs |
| CPU checks | Deployment forward, determinism, real-data backward and Qwen-head tests pass |
| Four scorer processes | Identical scores, exclusive temporary directories, no leftovers |
| Paths containing spaces | Scorer isolation smoke passes |
| Julia package | Fresh isolated Julia 1.10.5 environment installs and imports successfully |
| External dependency boundary | No bundled TSPDroneICP.jl or TSPDrone.jl source; pinned checkouts outside the repository |
| End-to-end Synthetic smoke | n=1,000, three search rounds, three exact repairs, feasible final route |
| End-to-end LaDe smoke | n=22, three search rounds, three exact repairs, feasible final route |
| Dependency-migration parity | Both fixed-work solution files are bitwise identical to the pre-migration version |
| Comparison code | External baseline experiment pipelines and weights excluded |
| Readme illustrations | Fig.1, Fig.2 and Fig.3 included |

The end-to-end smoke uses fixed work, not a new paper benchmark run. No full
training, Qwen fine-tuning or full benchmark suite was started during packaging.
The original experiment files and paper results were not overwritten.

The public distribution removes machine-specific metadata paths and excludes
server credentials and launch scripts. TSPDroneICP.jl and TSPDrone.jl sources
are not bundled. Setup obtains pinned upstream checkouts outside the repository;
the `ESNGSDM` adapter loads the original FI and partitioning implementations
from there, omitting unused Concorde/LKH/PyCall installation requirements.

Historical training states and supplementary recollected states are identified
separately in `docs/REPRODUCIBILITY.md`. The historical final model is supplied;
later supplementary and Qwen weights are retained outside the public release.
