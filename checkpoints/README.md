# Model artifacts

- `es_ngs_dm.pt`: final paper online deployment network, 205,253 parameters.
  The public file contains CPU tensors and portable metadata. All tensors are
  bitwise identical to the original `deployment_all.pt`.

This is the only distributed model weight. It was trained on the self-built
Synthetic instances, not on LaDe. Later supplementary students and Qwen LoRA
weights are not bundled. Qwen fine-tuning and decision-head source code remains
available in `repro/es_ngs/` and `repro/es_ngs/qwen_exp/scripts/`.

Source and released hashes are recorded in
`provenance/deployment_checkpoint.json` and `provenance/SHA256.json`.
Only load pickle-based `.pt` files from trusted sources. The supplied public
lightweight checkpoint also supports `torch.load(..., weights_only=True)`.
