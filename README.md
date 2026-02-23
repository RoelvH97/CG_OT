# TG-OT: Topology-guided CCTA-IVUS Registration via Optimal Transport Matching

This repository contains the implementation for the MICCAI 2026 submission:

**"TG-OT: Topology-guided CCTA-IVUS registration via optimal transport matching"**

## Overview

TG-OT provides a fully automatic framework for registering coronary CT angiography (CCTA) with intravascular ultrasound (IVUS), enabling comprehensive coronary analysis that neither modality can provide alone, **without requiring prior vessel segmentation**.

### Key Features

- **Segmentation-free registration**: Lightweight CNNs detect calcifications, bifurcations, and lumen radii directly on the topological (θ, z) cylinder, bypassing explicit segmentation that fails under IVUS acoustic shadowing from calcifications
- **Topologically coherent feature detection**: CNNs are supervised with a combined soft Dice and Betti matching loss, penalizing topological errors in predicted features for reliable downstream matching
- **Optimal transport feature matching**: An unbalanced Sinkhorn OT loss on the cylindrical geometry provides spatially informative gradients even for spatially disjoint predictions, complemented by a lumen matching term
- **Differentiable registration pipeline**: Centerline warping with a cumulative B-spline rotation parameterization naturally enforces smoothness and captures gradual catheter twist during IVUS pullback

### Method

CCTA and IVUS data are both transformed to a polar (r, θ, z) representation along an extracted coronary artery centerline. Frozen CNNs predict lumen radii, bifurcation locations, and calcification presence on the unwound topological cylinder for both modalities. Registration is then formulated as optimization over centerline warping parameters—global scaling and translation along the vessel axis, and per-frame local rotation and in-plane displacement—driven by an unbalanced Sinkhorn optimal transport loss that encodes geodesic cylindrical distance as transport cost, ensuring informative gradients throughout optimization.

## Example

Qualitative registration result showing cross-sectional normal vectors color-coded by cosine similarity, comparing pre-deformable alignment, Dice-only optimization, and the proposed OT+Dice optimization against the reference centerline:

![Registration example](assets/example.png)

## Repository Structure

```
TG_OT/
├── main_classify.py        # Train and evaluate the feature detection CNN
├── main_register.py        # Run CCTA-IVUS registration
├── configs/
│   ├── train_classifier.json
│   ├── eval_classifier.json
│   ├── register.json
│   └── eval_register.json
├── model/                  # FanCNN feature detection model and trainer
├── registration/           # Registration model and trainer
├── data/                   # Dataset and data module definitions
└── utils/                  # Polar transform, MPR, and I/O utilities
```

## Usage

### 1. Train the Feature Detection CNN

```bash
python main_classify.py configs/train_classifier.json
```

Set `"mode": "train"` in the config. Training runs 5-fold cross-validation automatically. Set `"modality"` to `"IVUS"` or `"MPR"` accordingly. Update `"root_dir"` in the config to point to your data directory.

### 2. Evaluate the Feature Detection CNN

```bash
python main_classify.py configs/eval_classifier.json
```

Set `"mode": "eval"` and provide checkpoint paths under `"ckpt"` in the model config.

### 3. Run Registration

```bash
python main_register.py configs/register.json
```

Set `"mode": "register"` and update `SAMPLE_IDS` in `main_register.py` with your case identifiers. Provide trained classifier checkpoints under the model config.

### 4. Evaluate Registration

```bash
python main_register.py configs/eval_register.json
```

Set `"mode": "eval"` to aggregate and statistically compare registration results across methods.

## License

© 2025 CC-BY 4.0
