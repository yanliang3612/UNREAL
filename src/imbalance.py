import torch


class Imbalance:
    """Create the class-imbalanced semi-supervised training split."""

    _DATASET_CONFIG = {
        "Cora": (20, 3),
        "CiteSeer": (20, 3),
        "PubMed": (20, 1),
    }

    def __init__(self, name, data, ratio) -> None:
        self.name = name
        self.data = data
        self.total_nodes = data.x.size(0)
        self.labels = data.y
        self.ratio = int(ratio)
        self.original_train_mask = data.train_mask.clone()
        self.num_classes = int(data.y.max().item()) + 1

        # Original public attribute names are kept as aliases.
        self.total_node = self.total_nodes
        self.label = self.labels
        self.data_train_mask = self.original_train_mask
        self.n_cls = self.num_classes

    def num_samples_per_class(self):
        training_labels = self.labels[self.original_train_mask]
        return [
            int((training_labels == class_index).sum().item())
            for class_index in range(self.num_classes)
        ]

    def target_samples_per_class(self):
        if self.name not in self._DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset: {self.name}")

        samples_per_class, num_imbalanced_classes = self._DATASET_CONFIG[self.name]
        first_imbalanced_class = self.num_classes - num_imbalanced_classes

        targets = []
        for class_index in range(self.num_classes):
            if self.ratio > 1 and class_index >= first_imbalanced_class:
                targets.append(int(samples_per_class / self.ratio))
            else:
                targets.append(samples_per_class)
        return targets

    def class_indices(self):
        node_indices = torch.arange(len(self.labels))
        return [
            node_indices[(self.labels == class_index) & self.original_train_mask]
            for class_index in range(self.num_classes)
        ]

    def split_semi_dataset(self):
        available_counts = self.num_samples_per_class()
        target_counts = self.target_samples_per_class()
        indices_by_class = self.class_indices()

        selected_by_class = []
        train_mask = indices_by_class[0].new_zeros(
            self.total_nodes,
            dtype=torch.bool,
        )

        for class_index in range(self.num_classes):
            class_indices = indices_by_class[class_index]
            target_count = target_counts[class_index]

            if available_counts[class_index] > target_count:
                permutation = torch.randperm(len(class_indices))
                selected_indices = class_indices[permutation][:target_count]
            else:
                selected_indices = class_indices

            selected_by_class.append(selected_indices)
            train_mask[selected_indices] = True

        expected_count = sum(target_counts)
        assert train_mask.sum().long() == expected_count
        assert sum(len(indices) for indices in selected_by_class) == expected_count
        return train_mask

    # Backward-compatible method names from the original implementation.
    n_data = num_samples_per_class
    class_num_list = target_samples_per_class
    get_idx_info = class_indices
