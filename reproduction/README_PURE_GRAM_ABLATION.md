# Pure Gram loss ablation (100,000 epochs)

Seven comparable experiments retain the original `workingone` Gram and diagonal loss formulations. Only the three component switches change. Both auxiliary reconstruction consistency and entropy weights are fixed at zero across all seven configurations.

```text
TOTAL = 10*(w_l21*diag_l21 + 10*w_sam*diag_sam)
      + 100*(w_l21*gram_l21 + w_ergas*gram_ergas
             + 10*w_sam*gram_sam + w_sam*gram_sam_transpose)
```

## Requirements

The local UISSF project must include `scripts/uissf_cli.py` with `--reconstruction-weight` and `--entropy-weight` overrides, `scripts/evaluate_checkpoint_comparison.py`, prepared hyperspectral dataset inputs, original SRF, and decomposition checkpoints. The generator does not include private datasets or model weights.

## Generate scripts without running experiments

```bash
cd /trinity/home/jizh/UISSF_release && PYTHONPATH=. /trinity/home/jizh/venvs/nerfstudio/bin/python reproduction/build_experiments.py --dataset chikusei --phase loss --epochs 100000 --eval-every 1000 --save-every 1000 --log-every 100 --pure-gram-loss --output-root /trinity/home/jizh/UISSF_runs/manuscript_pure_gram_100k --check-inputs
```

Without `--submit`, this **only prepares seven Slurm scripts** and a manifest; it does not start training.

## Study design

Configurations: l21, ergas, sam, l21_ergas, l21_sam, ergas_sam, all. Each starts with the same mapper seed, runs with the same original coefficients and optimizer schedule, and uses a separate result folder. The auxiliary-free study is a controlled diagnostic, not an identical reproduction of the original manuscript's auxiliary-inclusive objective.

The external WLRTR, u²-MDN and EDIP-Net baseline implementations are not part of this patch.
