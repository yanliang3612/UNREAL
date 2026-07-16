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

---

## 1. Introduction

<p align="center">
  <img src="figure/unreal_logo.png" width="90%" alt="UNREAL: Geometric Imbalance" />
</p>

Class imbalance in graph data presents a significant challenge for effective node classification, particularly in semi-supervised scenarios. In this work, we formally introduce the concept of geometric imbalance, which captures how message passing on class-imbalanced graphs leads to geometric ambiguity among minority-class nodes in the Riemannian manifold embedding space. We provide a rigorous theoretical analysis of geometric imbalance on the Riemannian manifold and propose a unified framework that explicitly mitigates it through pseudo-label alignment, node reordering, and ambiguity filtering. Extensive experiments on diverse benchmarks show that our approach consistently outperforms existing methods, especially under severe class imbalance. Our findings offer new theoretical insights and practical tools for robust semi-supervised node classification.

## 2. Environment

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

## 3. Training Hyperparameters

### 3.1 Cora-Semi (imbalance ratio= 10, 20, 50, 100)

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

### 3.2 CiteSeer-Semi (imbalance ratio= 10, 20, 50, 100)

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

### 3.3 PubMed-Semi (imbalance ratio= 10, 20, 50, 100)

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


## 4. Baselines

Baseline implementations and hyperparameter configurations:

- For the implementation and hyperparameters setting of **Re-Weight, PC Softmax, BalancedSoftmax, TAM**, please refer to [TAM](https://github.com/Jaeyun-Song/TAM).
- For the implementation and hyperparameters setting of **GraphSmote**, please refer to [GraphSmote](https://github.com/TianxiangZhao/GraphSmote).
- For the implementation and hyperparameters setting of **Renode**, please refer to [Renode](https://github.com/victorchen96/ReNode).
- For the implementation and hyperparameters setting of **GraphENS**, please refer to [GraphENS](https://github.com/JoonHyung-Park/GraphENS).

We strictly adhere to the hyperparameter settings as specified in these papers. For detailed information, please refer to the respective publications.

## 5. Configuration

All algorithms and models are implemented in Python and PyTorch Geometric. Most experiments are conducted on a server equipped with an NVIDIA GeForce RTX 3090 GPU (24 GB GDDR6X memory) and an Intel(R) Xeon(R) Silver 4210R CPU @ 2.40 GHz. The experiments on **ogbn-arxiv** and **Flickr** are conducted using an NVIDIA A100 80GB PCIe Tensor Core GPU (80 GB HBM2e memory).

## 6. Cite Us

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
