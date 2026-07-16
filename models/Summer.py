from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.cluster import KMeans as SklearnKMeans
from torch import nn
from torch.optim import Adam

from embedder import Embedder
from layers import GNN, Classifier
from src.data import Planetoid
from src.rbo import rbo_score
from src.utils import random_seed, reset, set_random_seeds


class SummerTrainer(Embedder):
    PRETRAIN_EPOCHS = 200
    SUPPORTED_DATASETS = {"Cora", "CiteSeer", "PubMed"}

    def _init_model(self) -> None:
        self.model = Summer(self.encoder, self.classifier).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.decay,
        )

    def _run_kmeans(self, representations):
        """Cluster normalized embeddings with the configured backend."""
        if self.args.kmeans_backend == "cpu":
            representations_cpu = representations.cpu().numpy()
            clustering = SklearnKMeans(
                n_clusters=self.args.num_K,
                n_init=10,
            ).fit(representations_cpu)
            cluster_labels = clustering.predict(representations_cpu)
            return cluster_labels, representations_cpu

        if not representations.is_cuda:
            raise RuntimeError(
                "GPU K-means requires a CUDA-enabled PyTorch installation "
                "and an available GPU."
            )

        try:
            from torch_kmeans import KMeans as TorchKMeans
        except ImportError as exc:
            raise ImportError(
                "GPU K-means requires torch-kmeans. Install it with "
                "`python -m pip install torch-kmeans==0.2.0`."
            ) from exc

        clustering = TorchKMeans(
            n_clusters=self.args.num_K,
            init_method="k-means++",
            num_init=10,
            max_iter=300,
            tol=1e-4,
            seed=int(torch.initial_seed() % (2**31)),
            verbose=False,
        )
        with torch.no_grad():
            cluster_labels = clustering(representations.unsqueeze(0)).labels.squeeze(0)

        return (
            cluster_labels.cpu().numpy(),
            representations.cpu().numpy(),
        )

    def _init_dataset(self) -> None:
        if self.args.dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {self.args.dataset}. "
                f"Choose one of {sorted(self.SUPPORTED_DATASETS)}."
            )

        self.data = Planetoid(
            self.path,
            self.args.dataset,
            transform=T.NormalizeFeatures(),
            split="public",
            ratio=self.args.imb_ratio,
        )[0].to(self.device)

        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)

        # This is algebraically equivalent to summing a dense adjacency matrix,
        # but avoids an O(num_nodes^2) allocation.
        edge_count = torch.tensor(
            self.data.edge_index.size(1),
            dtype=torch.float32,
            device=self.device,
        )
        average_degree = edge_count / self.data.num_nodes
        eta = self.data.num_nodes / average_degree ** len(self.hidden_layers)

        label_counts = self.labels[self.train_mask].unique(return_counts=True)[1]
        additions = (label_counts * 3 * eta / len(self.labels[self.train_mask])).to(
            torch.int64
        )
        self.t = additions / self.args.rounds
        self.t[:4] = self.args.ad
        self.t[-3:] = self.args.ad

    def _build_cluster_pseudo_labels(
        self,
        cluster_labels,
        representations,
    ):
        labels_cpu = self.labels.cpu()
        train_mask_cpu = self.running_train_mask.cpu()

        labeled_centroids = []
        for class_index in range(self.num_classes):
            class_mask = ((labels_cpu == class_index) & train_mask_cpu).numpy()
            labeled_centroids.append(representations[class_mask].mean(axis=0))
        labeled_centroids = np.stack(labeled_centroids)

        pseudo_labels = np.full_like(cluster_labels, fill_value=-1)
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) != self.args.num_K:
            print("Warning: K-means produced one or more empty clusters.")

        unlabeled_mask = ~train_mask_cpu.numpy()
        for cluster_index in unique_clusters:
            cluster_mask = (cluster_labels == cluster_index) & unlabeled_mask
            if not cluster_mask.any():
                continue

            cluster_centroid = representations[cluster_mask].mean(axis=0)
            distances = ((labeled_centroids - cluster_centroid) ** 2).sum(axis=1)
            pseudo_labels[cluster_mask] = np.argmin(distances)

        assert (pseudo_labels[unlabeled_mask] == -1).sum() == 0
        return pseudo_labels, labeled_centroids

    def pretrain(self, repetition: int, round_index: int) -> None:
        """Run one UNREAL self-training round."""
        for epoch in range(self.PRETRAIN_EPOCHS):
            self.model.train()
            self.optimizer.zero_grad()

            logits, _ = self.model.cls(self.data)
            loss = F.cross_entropy(
                logits[self.running_train_mask],
                self.labels[self.running_train_mask],
            )
            loss.backward()
            self.optimizer.step()

            print(
                f"[Repetitions: {repetition + 1}]"
                f"[Rounds: {round_index + 1}/{self.args.rounds}]"
                f"[Epoch: {epoch + 1}/{self.PRETRAIN_EPOCHS}] "
                f"Loss: {loss.item():.4f}"
            )

        if not self.args.clustering:
            raise RuntimeError("UNREAL requires clustering to generate pseudo-labels.")

        self.model.eval()
        with torch.no_grad():
            representations = F.normalize(
                self.model.encoder(self.data),
                dim=1,
            )
        cluster_labels, representations = self._run_kmeans(representations)
        pseudo_labels, labeled_centroids = self._build_cluster_pseudo_labels(
            cluster_labels,
            representations,
        )

        with torch.no_grad():
            logits, _ = self.model.cls(self.data)
            predictions = F.softmax(logits, dim=1)

        pseudo_targets, self.running_train_mask = self.UNREAL(
            predictions,
            pseudo_labels,
            representations,
            labeled_centroids,
        )
        self.labels[self.running_train_mask] = torch.argmax(
            pseudo_targets[self.running_train_mask],
            dim=1,
        )

    def train(self) -> None:
        for repetition in range(self.args.repetitions):
            set_random_seeds(random_seed(repetition, self.args.dataset))
            self._init_dataset()

            input_size = self.data.x.size(1)
            representation_size = self.hidden_layers[-1]
            self.unique_labels = self.data.y.unique()
            self.num_classes = len(self.unique_labels)

            self.encoder = GNN(
                [input_size] + self.hidden_layers,
                net=self.args.net,
                n_heads=self.args.n_head,
                chebyshev_order=self.args.chebgcn_para,
            )
            self.classifier = Classifier(
                representation_size,
                self.num_classes,
            )

            for round_index in range(self.args.rounds):
                self._init_model()
                self.pretrain(repetition, round_index)

            for epoch in range(1, self.args.epochs + 1):
                self.model.train()
                self.optimizer.zero_grad()

                logits, _ = self.model.cls(self.data)
                loss = F.cross_entropy(
                    logits[self.running_train_mask],
                    self.labels[self.running_train_mask],
                )
                loss.backward()
                self.optimizer.step()

                status = (
                    f"[Repetitions: {repetition + 1}]"
                    f"[Epoch: {epoch}/{self.args.epochs}] "
                    f"Loss: {loss.item():.4f} "
                )
                self.evaluate(self.data, status)
                if self.cnt == self.args.patience:
                    print("Early stopping.")
                    break

            self.save_results(repetition)

        self.summary()

    def UNREAL(
        self,
        predictions,
        pseudo_labels,
        representations,
        labeled_centroids,
    ):
        classifier_labels = torch.argmax(predictions, dim=1)
        classifier_labels_cpu = classifier_labels.cpu().numpy()
        confidence_cpu = torch.max(predictions, dim=1)[0].cpu().numpy()
        running_train_mask_cpu = self.running_train_mask.cpu().numpy()

        pseudo_targets = F.one_hot(self.labels).float()
        pseudo_targets[~self.running_train_mask] = 0
        num_classes = pseudo_targets.shape[1]
        assert len(self.t) >= num_classes

        candidate_indices = [[] for _ in range(num_classes)]
        for node_index, classifier_label in enumerate(classifier_labels_cpu):
            if (
                not running_train_mask_cpu[node_index]
                and pseudo_labels[node_index] == classifier_label
            ):
                candidate_indices[int(classifier_label)].append(node_index)

        ranking_indices = []
        for class_index, class_candidates in enumerate(candidate_indices):
            class_representations = representations[class_candidates]
            distances = (
                (class_representations - labeled_centroids[class_index]) ** 2
            ).sum(axis=1)
            confidences = confidence_cpu[class_candidates]

            confidence_order = confidences.argsort()[::-1]
            confidence_ranks = confidence_order.argsort()
            distance_order = distances.argsort()
            distance_ranks = distance_order.argsort()

            rbo = rbo_score(
                confidence_ranks,
                distance_ranks,
                self.args.rbo,
            )
            if rbo >= 0.5:
                combined_ranks = rbo * distance_ranks + (1 - rbo) * confidence_ranks
            else:
                combined_ranks = (1 - rbo) * distance_ranks + rbo * confidence_ranks
            ranking_indices.append(combined_ranks.argsort())

        selected_indices = []
        selected_counts = [0] * num_classes

        for class_index in range(num_classes):
            ordered_candidates = np.asarray(candidate_indices[class_index])[
                ranking_indices[class_index]
            ]
            for node_index in ordered_candidates:
                if selected_counts[class_index] >= self.t[class_index]:
                    break

                node_distances = (
                    (representations[node_index] - labeled_centroids) ** 2
                ).sum(axis=1)
                nearest_classes = node_distances.argsort()
                nearest = nearest_classes[0]
                second_nearest = nearest_classes[1]
                distance_margin = (
                    node_distances[second_nearest] - node_distances[nearest]
                ) / node_distances[nearest]

                if distance_margin > self.args.threshold:
                    selected_indices.append(int(node_index))
                    selected_counts[class_index] += 1

        indicator = torch.zeros_like(self.running_train_mask)
        indicator[selected_indices] = True
        indicator &= ~self.running_train_mask

        hard_predictions = torch.zeros_like(predictions)
        node_indices = torch.arange(
            len(classifier_labels),
            device=self.device,
        )
        hard_predictions[node_indices, classifier_labels] = 1.0
        hard_predictions[self.running_train_mask] = pseudo_targets[
            self.running_train_mask
        ]

        updated_targets = deepcopy(pseudo_targets)
        updated_train_mask = deepcopy(self.running_train_mask)
        updated_train_mask[indicator] = True
        updated_targets[indicator] = hard_predictions[indicator]
        return updated_targets, updated_train_mask


class Summer(nn.Module):
    def __init__(self, encoder, classifier) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.reset_parameters()

    def forward(self, data):
        embeddings = self.encoder(data)
        return self.classifier(embeddings)

    def cls(self, data):
        return self.forward(data)

    def reset_parameters(self) -> None:
        reset(self.encoder)
        reset(self.classifier)


# Backward-compatible class name used by the original entry point.
Summer_Trainer = SummerTrainer
