# Data

- `synthetic/manifest_all.csv`: 900 frozen geometries, with original splits
  and SHA256 values. The paper evaluation list contains 30 n=1,000 cases.
- `lade/manifest.csv`: eight demand-grounded external geometries. CSV local
  coordinates and Julia serialized geometry are both supplied.
- `training/`: 480 supplementary states from 40 independently generated training
  instances, exact outcomes, soft labels and all-data offline preference utilities.
- `splits/`: exact training-instance and evaluation lists, plus five fixed
  instance-grouped training/validation splits.

Within each state directory, only `X.npy`, `A.npy`, `action.npy`, `global.npy`
and `mask.npy` are online model inputs. `y*.npy` and `feas.npy` are supervision.
The candidate's exact feasibility label is never supplied as an input feature.
The mask marks valid local tokens, not candidate feasibility.

Supplementary all-data utilities must not be reused as held-out-fold teachers.
Use `scripts/train.py --fold N` to train and export a teacher within each fold.

The raw parcel dataset is obtained from the official LaDe owner. Released
external scenarios use centered AOI coordinates and synthetic station layouts.
See `THIRD_PARTY_NOTICES.md` for data licensing and `docs/REPRODUCIBILITY.md`
for generation and artifact identity.
