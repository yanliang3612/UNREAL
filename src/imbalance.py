import torch

class Imbalance:

    def __init__(self, name,data, ratio):
        self.name = name
        self.data = data
        self.total_node = len(data.x)
        self.label = data.y
        self.ratio = int(ratio)
        self.data_train_mask = data.train_mask.clone()
        self.n_cls = data.y.max().item() + 1


    def n_data(self):
        n_data = []
        stats = self.data.y[self.data_train_mask]
        for i in range(self.n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        return n_data


    def class_num_list(self):
        class_num_list = []
        if self.name == 'Cora':
            class_sample_num = 20
            imb_class_num = 3
        elif self.name == 'CiteSeer':
            class_sample_num = 20
            imb_class_num = 3
        elif self.name == 'PubMed':
            class_sample_num = 20
            imb_class_num = 1
        else:
            print("no this dataset: {args.dataset}")
        for i in range(self.n_cls):
            if self.ratio > 1 and i > self.n_cls - 1 - imb_class_num:  # only imbalance the last classes
                class_num_list.append(int(class_sample_num * (1. / self.ratio)))
            else:
                class_num_list.append(class_sample_num)

        return class_num_list



    def get_idx_info(self):
        index_list = torch.arange(len(self.label))
        idx_info = []
        for i in range(self.n_cls):
            cls_indices = index_list[((self.label == i) & self.data_train_mask)]
            idx_info.append(cls_indices)
        return idx_info


    def split_semi_dataset(self):
        n_data = self.n_data()
        class_num_list = self.class_num_list()
        idx_info = self.get_idx_info()
        new_idx_info = []
        _train_mask = idx_info[0].new_zeros(self.total_node, dtype=torch.bool)
        for i in range(self.n_cls):
            if n_data[i] > class_num_list[i]:
                cls_idx = torch.randperm(len(idx_info[i]))
                cls_idx = idx_info[i][cls_idx]
                cls_idx = cls_idx[:class_num_list[i]]
                new_idx_info.append(cls_idx)
            else:
                new_idx_info.append(idx_info[i])
            _train_mask[new_idx_info[i]] = True
        assert _train_mask.sum().long() == sum(class_num_list)
        assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

        return _train_mask