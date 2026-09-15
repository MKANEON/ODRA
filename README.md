<div align="center">

# ODRA

### Orthogonal Decomposition-Based Representation Augmentation for Imbalanced ERP-BCI Decoding

Jiayi Li<sup>1</sup>, Mingming Yang<sup>1</sup>, Hao Wang<sup>1</sup>, Simiao Li<sup>1</sup>, Baolian Shan<sup>1</sup>, Minpeng Xu<sup>1,2,*</sup>, Jiayuan Meng<sup>1,2,*</sup>, and Dong Ming<sup>1,2</sup>

<sup>1</sup> Academy of Medical Engineering and Translational Medicine, Tianjin University, Tianjin 300072, China<br>
<sup>2</sup> Haihe Laboratory of Brain-Computer Interaction and Human-Machine Integration, Tianjin 300392, China<br>
<sup>*</sup> Corresponding authors: Minpeng Xu (minpeng.xu@tju.edu.cn) and Jiayuan Meng (mengjiayuan@tju.edu.cn)

</div>

## Overview

This repository provides the core implementation and a minimal usage example of orthogonal decomposition-based representation augmentation (ODRA) for imbalanced ERP-BCI decoding. ODRA performs augmentation in the learned representation space by recombining class-specific components from minority-class features with class-general components from training trials. The trained backbone remains fixed during representation augmentation and classifier head fine-tuning.

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

