<div align="center">

# ODRA

### Orthogonal Decomposition-Based Representation Augmentation for Imbalanced ERP-BCI Decoding

Jiayi Li, Mingming Yang, Hao Wang, Simiao Li, Baolian Shan, Minpeng Xu, Jiayuan Meng, and Dong Ming

</div>

## Overview

This repository provides the core implementation and a minimal usage example of orthogonal decomposition-based representation augmentation (ODRA) for imbalanced ERP-BCI decoding. ODRA performs augmentation in the learned representation space by combining minority-class discriminative projections with complementary components from training trials. The trained backbone remains fixed during representation augmentation and classifier-head fine-tuning.

The current release includes an EEGNet-based implementation and a minimal single-subject stratified five-fold cross-validation example.

## Quick Start

Install the required packages:

```bash
pip install numpy torch scikit-learn
```

Prepare preprocessed NumPy arrays with `X.shape == (samples, channels, time)` and `y.shape == (samples,)`, then run:

```bash
python run_example_5cv.py \
  --x path/to/X.npy \
  --y path/to/y.npy \
  --output results/odra_5cv_summary.json
```

The provided script is a minimal usage example based on stratified five-fold cross-validation; it does not reproduce every dataset-specific evaluation protocol reported in the manuscript.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{li2026odra,
  author = {Jiayi Li and Mingming Yang and Hao Wang and Simiao Li and Baolian Shan and Minpeng Xu and Jiayuan Meng and Dong Ming},
  title  = {Orthogonal Decomposition-Based Representation Augmentation for Imbalanced ERP-BCI Decoding},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

