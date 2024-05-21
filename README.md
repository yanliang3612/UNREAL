# UNREAL: Unlabeled Nodes Retrieval and Labeling for Heavily-imbalanced Node Classification. (Arxiv 2023)

## Introduction

Official Pytorch implementation of Arxiv 2023 paper "[UNREAL:Unlabeled Nodes Retrieval and Labeling for Heavily-imbalanced Node Classification](https://arxiv.org/abs/2303.10371)"

![unreal](figure/final.png)
Extremely skewed label distributions are common in real-world node classification tasks. If not dealt with appropriately, it significantly hurts the performance of GNNs in minority classes. Due to its practical importance, there have been a series of recent research devoted to this challenge. Existing over-sampling techniques smooth the label distribution by generating ``fake'' minority nodes and synthesizing their features and local topology, which largely ignore the rich information of unlabeled nodes on graphs. In this paper, we propose UNREAL, an iterative over-sampling method. The first key difference is that we only add unlabeled nodes instead of synthetic nodes, which eliminates the challenge of feature and neighborhood generation. To select which unlabeled nodes to add, we propose geometric ranking to rank unlabeled nodes. Geometric ranking exploits unsupervised learning in the node embedding space to effectively calibrates pseudo-label assignment. Finally, we identify the issue of geometric imbalance in the embedding space and provide a simple metric to filter out geometrically imbalanced nodes. Extensive experiments on real-world benchmark datasets are conducted, and the empirical results show that our method significantly outperforms current state-of-the-art methods consistent on different datasets with different imbalance ratios.

## Environment
```bash
conda create -n "unreal" python=3.8.13
source activate unreal
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.3.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
```
## Training Hyperparameters
### Cora-Semi (imbalance ratio= 10, 20, 50, 100)
- Cora-GCN (imbalance ratio= 10)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 10 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 10)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 10 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 10)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 10 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GCN (imbalance ratio= 20)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 20 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 20)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 20 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 20)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 20 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GCN (imbalance ratio= 50)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 50 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 50)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 50 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 50)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 50 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GCN (imbalance ratio= 100)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 100 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-GAT (imbalance ratio= 100)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 100 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- Cora-SAGE (imbalance ratio= 100)
  ```bash
  python main.py --dataset Cora --repetitions 5 --imb_ratio 100 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```

### CiteSeer-Semi (imbalance ratio= 10, 20, 50, 100)
- CiteSeer-GCN (imbalance ratio= 10)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 10 --net GCN  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 10)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 10 --net GAT  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 10)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 10 --net SAGE  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GCN (imbalance ratio= 20)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 20 --net GCN  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 20)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 20 --net GAT  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 20)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 20 --net SAGE  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GCN (imbalance ratio= 50)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 50 --net GCN  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 50)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 50 --net GAT  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 50)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 50 --net SAGE  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GCN (imbalance ratio= 100)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 100 --net GCN  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-GAT (imbalance ratio= 100)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 100 --net GAT  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```
- CiteSeer-SAGE (imbalance ratio= 100)
  ```bash
  python main.py --dataset CiteSeer --repetitions 5 --imb_ratio 100 --net SAGE  --rounds 10 --ad 5 --rbo 0.5 --threshold 0.25
  ```

### PubMed-Semi (imbalance ratio= 10, 20, 50, 100)
- PubMed-GCN (imbalance ratio= 10)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 10 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 10)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 10 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 10)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 10 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GCN (imbalance ratio= 20)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 20 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 20)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 20 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 20)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 20 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GCN (imbalance ratio= 50)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 50 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 50)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 50 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 50)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 50 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GCN (imbalance ratio= 100)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 100 --net GCN  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-GAT (imbalance ratio= 100)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 100 --net GAT  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```
- PubMed-SAGE (imbalance ratio= 100)
  ```bash
  python main.py --dataset  PubMed --repetitions 5 --imb_ratio 100 --net SAGE  --rounds 40 --ad 4 --rbo 0.5 --threshold 0.25
  ```


## Baselines
### The Implementation of Baselines and the Configuration of Hyperparameters
- For the implementation and hyperparameters setting of **Re-Weight, PC Softmax, BalancedSoftmax, TAM**, please refer to [TAM](https://github.com/Jaeyun-Song/TAM).
- For the implementation and hyperparameters setting of **GraphSmote**, please refer to [GraphSmote](https://github.com/TianxiangZhao/GraphSmote).
- For the implementation and hyperparameters setting of **Renode**, please refer to [Renode](https://github.com/victorchen96/ReNode).
- For the implementation and hyperparameters setting of **GraphENS**, please refer to [GraphENS](https://github.com/JoonHyung-Park/GraphENS).

We strictly adhere to the hyperparameter settings as specified in these papers. For detailed information, please refer to the respective publications.







## Configuration
All the algorithms and models are implemented in Python and PyTorch Geometric. Experiments are
conducted on a server with an NVIDIA 3090 GPU (24 GB memory) and an Intel(R) Xeon(R) Silver
4210R CPU @ 2.40GHz.
