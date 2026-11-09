# UISSF — Registration-Free Hyperspectral Reconstruction from RGB

**Paper:** *Registration-Free Hyperspectral Reconstruction from RGB via a Permutation-Invariant Gram-Matrix Principle*

**Authors:** Jiangsan Zhao, Masayuki Hirafuji, Seishi Ninomiya, Jakob Geipel, Wei Guo

**Preprint:** [arXiv:2608.14994](https://arxiv.org/abs/2608.14994) · [DOI: 10.48550/arXiv.2608.14994](https://doi.org/10.48550/arXiv.2608.14994) (August 2026)

**Journal manuscript:** Submitted to **IEEE Transactions on Multimedia (TMM)**. The arXiv manuscript is a preprint; journal acceptance or publication is not implied.

## Overview

UISSF addresses registration-free hyperspectral reconstruction from high-resolution RGB and low-resolution hyperspectral observations. Its permutation-invariant abundance Gram-matrix principle allows learning without paired, spatially aligned RGB–HSI pixels or a predefined camera response function. The method uses a residual spectral mapping and pretrained decomposition.

## Reproduction scripts

The repository is under active development. Currently available:

- [Experiment script generator](reproduction/build_experiments.py): generates independent Slurm jobs for Chikusei loss ablations, plus other study phases.
- [Pure Gram-loss ablation notes](reproduction/README_PURE_GRAM_ABLATION.md): zero-weight controlled ablation of ℓ2,1, ERGAS, and SAM components.
- [Test for ablation generation](tests/test_pure_gram_suite.py).

The generated jobs are configured for the original research HPC environment and require the full model implementation, prepared datasets, SRF, and pretrained decomposition checkpoints. **The experiment generator alone is not a complete standalone implementation**; the full reproduction code and dependencies will be documented and released separately as available. No training is started by generating scripts unless `--submit` is explicitly supplied.

## Citation

If you use the preprint or research concept, please cite:

```bibtex
@misc{zhao2026registrationfree,
  title        = {Registration-Free Hyperspectral Reconstruction from RGB via a Permutation-Invariant Gram-Matrix Principle},
  author       = {Zhao, Jiangsan and Hirafuji, Masayuki and Ninomiya, Seishi and Geipel, Jakob and Guo, Wei},
  year         = {2026},
  eprint       = {2608.14994},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2608.14994},
  url          = {https://arxiv.org/abs/2608.14994}
}
```

## Status

Preprint available since August 2026; peer-reviewed journal submission is under consideration at IEEE Transactions on Multimedia. Please refer to the preprint for methodological and experimental details. The code repository remains a work in progress.
