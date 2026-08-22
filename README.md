<div align="center">
  <img src="figure/unreal_header_logo.png" width="150" alt="UNREAL logo" />
</div>

<h1 align="center">Geometric Imbalance in Semi-Supervised Node Classification</h1>

<p align="center">
  <a href="https://divinyan.com/UNREAL/"><img src="https://img.shields.io/badge/Project-Page-2F80ED" alt="Project Page" /></a>
  <a href="https://openreview.net/forum?id=BND9CutZf6"><img src="https://img.shields.io/badge/Paper-NeurIPS_2025-B31B1B" alt="NeurIPS 2025 Paper" /></a>
  <a href="https://arxiv.org/abs/2303.10371"><img src="https://img.shields.io/badge/arXiv-2303.10371-B31B1B" alt="arXiv" /></a>
</p>

<p align="center">
  The official PyTorch implementation of the NeurIPS 2025 paper
  <em>Geometric Imbalance in Semi-Supervised Node Classification</em>.
</p>

<p align="center">
  Liang Yan, Shengzhong Zhang, Bisheng Li, Menglin Yang, Chen Yang,<br />
  Min Zhou, Weiyang Ding, Yutong Xie, Zengfeng Huang
</p>

<p align="center">
  <strong>The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS), 2025</strong>
</p>

<p align="center">
  <strong>Previous version:</strong>
  <em>UNREAL: Unlabeled Nodes Retrieval and Labeling for Heavily-Imbalanced Node Classification</em> (arXiv 2023)
</p>

<p align="center">
  <a href="mailto:yanliangfdu@gmail.com">
    <img src="https://readme-typing-svg.demolab.com?font=Inter&amp;weight=700&amp;size=18&amp;pause=1200&amp;color=C56A4A&amp;center=true&amp;vCenter=true&amp;width=860&amp;lines=Real-time+Q%26A+%E2%80%A2+code+contributions+%E2%80%A2+pull+requests;contact%3A+yanliangfdu%40gmail.com" alt="Real-time Q&amp;A, code contributions, pull requests, and contact" />
  </a>
</p>

<p align="center">
  <img src="figure/unreal_logo.png" width="90%" alt="UNREAL: Geometric Imbalance" />
</p>

## Overview

UNREAL addresses a failure mode of GNN self-training on class-imbalanced graphs: minority-class nodes can become geometrically ambiguous in the embedding space, making their pseudo-labels unreliable. The framework improves pseudo-label quality by aligning clustering and classification predictions, prioritizing candidates using both geometric proximity and confidence, and filtering ambiguous nodes before retraining.

## 1. Environment

The setup script targets Linux x86_64 with CUDA 11.3 and creates a Conda environment named `unreal`:

```bash
bash scripts/setup_env.sh
conda activate unreal
```

To use a different environment name:

```bash
ENV_NAME=my_unreal_env bash scripts/setup_env.sh
conda activate my_unreal_env
```

The repository includes both the raw and processed Cora files under
`data/Cora`, so Cora experiments can run immediately after cloning and setting
up the environment, without downloading the dataset separately.

### Optional GPU-Accelerated K-means

The reported configuration uses scikit-learn K-means on the CPU, which remains the default:

```bash
python main.py \
  --dataset Cora \
  --kmeans_backend cpu
```

For large datasets, K-means can instead run on an NVIDIA GPU through `torch-kmeans`:

```bash
python main.py \
  --dataset Cora \
  --device cuda:7 \
  --kmeans_backend gpu
```

The setup script installs `torch-kmeans==0.2.0`. For an existing environment, install it manually:

```bash
python -m pip install torch-kmeans==0.2.0
```

The GPU option accelerates the clustering stage only and requires CUDA. Because the CPU and GPU backends use different K-means implementations, their cluster assignments may differ slightly even with equivalent optimization settings. Omit `--kmeans_backend` to reproduce the default CPU configuration.

### Selecting a Compute Device

Use `--device` to select a physical GPU directly, without remapping GPU indices:

```bash
python main.py \
  --dataset Cora \
  --device cuda:7
```

The default value is `auto`, which uses `cuda:0` when CUDA is available and otherwise falls back to the CPU. You can also explicitly use `--device cpu`.

## 2. Training Hyperparameters

### 2.1 Cora-Semi (imbalance ratio= 10, 20, 50, 100)

- Cora-GCN (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 10 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 10 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 10 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GCN (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 20 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 20 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 20 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GCN (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 50 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 50 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 50 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GCN (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 100 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 100 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset Cora \
    --repetitions 5 \
    --imb_ratio 100 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```

### 2.2 CiteSeer-Semi (imbalance ratio= 10, 20, 50, 100)

- CiteSeer-GCN (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 10 \
    --net GCN \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 10 \
    --net GAT \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 10 \
    --net SAGE \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GCN (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 20 \
    --net GCN \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 20 \
    --net GAT \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 20 \
    --net SAGE \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GCN (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 50 \
    --net GCN \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 50 \
    --net GAT \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 50 \
    --net SAGE \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GCN (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 100 \
    --net GCN \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 100 \
    --net GAT \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset CiteSeer \
    --repetitions 5 \
    --imb_ratio 100 \
    --net SAGE \
    --rounds 10 \
    --ad 5 \
    --rbo 0.5 \
    --threshold 0.25
  ```

### 2.3 PubMed-Semi (imbalance ratio= 10, 20, 50, 100)

- PubMed-GCN (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 10 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 10 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 10)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 10 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GCN (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 20 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 20 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 20)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 20 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GCN (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 50 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 50 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 50)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 50 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GCN (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 100 \
    --net GCN \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 100 \
    --net GAT \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 100)
  ```bash
  python main.py \
    --dataset PubMed \
    --repetitions 5 \
    --imb_ratio 100 \
    --net SAGE \
    --rounds 40 \
    --ad 4 \
    --rbo 0.5 \
    --threshold 0.25
  ```


## 3. Baselines

Baseline implementations and hyperparameter configurations:

- For the implementation and hyperparameters setting of **Re-Weight, PC Softmax, BalancedSoftmax, TAM**, please refer to [TAM](https://github.com/Jaeyun-Song/TAM).
- For the implementation and hyperparameters setting of **GraphSmote**, please refer to [GraphSmote](https://github.com/TianxiangZhao/GraphSmote).
- For the implementation and hyperparameters setting of **Renode**, please refer to [Renode](https://github.com/victorchen96/ReNode).
- For the implementation and hyperparameters setting of **GraphENS**, please refer to [GraphENS](https://github.com/JoonHyung-Park/GraphENS).

We strictly adhere to the hyperparameter settings as specified in these papers. For detailed information, please refer to the respective publications.

## 4. Configuration

All algorithms and models are implemented in Python and PyTorch Geometric. Most experiments are conducted on a server equipped with an NVIDIA GeForce RTX 3090 GPU (24 GB GDDR6X memory) and an Intel(R) Xeon(R) Silver 4210R CPU @ 2.40 GHz. The experiments on **ogbn-arxiv** and **Flickr** are conducted using an NVIDIA A100 80GB PCIe Tensor Core GPU (80 GB HBM2e memory).

## 5. Cite Us

If you find this work useful, please cite:

```bibtex
@inproceedings{yan2025geometric,
  title     = {Geometric Imbalance in Semi-Supervised Node Classification},
  author    = {Yan, Liang and Zhang, Shengzhong and Li, Bisheng and Yang, Mengling and Yang, Chen and Zhou, Min and Ding, Weiyang and Xie, Yutong and Huang, Zengfeng},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=BND9CutZf6}
}
```
