import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset, Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.class_indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.class_indices[int(label)].append(index)
        self.classes = list(self.class_indices.keys())
        self.total_samples = len(self.labels)
        self.base_per_class = self.batch_size // max(len(self.classes), 1)
        self.remain_samples = self.batch_size % max(len(self.classes), 1)

    def _init_class_iterator(self, cls):
        indices = self.class_indices[cls].copy()
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __iter__(self):
        class_iters = {cls: self._init_class_iterator(cls) for cls in self.classes}
        for _ in range(len(self)):
            batch = []
            for cls in self.classes:
                for _ in range(self.base_per_class):
                    try:
                        batch.append(next(class_iters[cls]))
                    except StopIteration:
                        class_iters[cls] = self._init_class_iterator(cls)
                        batch.append(next(class_iters[cls]))
            if self.remain_samples > 0:
                extras = []
                for indices in self.class_indices.values():
                    extras.extend(indices)
                if self.shuffle:
                    np.random.shuffle(extras)
                batch.extend(extras[: self.remain_samples])
            yield batch[: self.batch_size]

    def __len__(self):
        return int(np.ceil(self.total_samples / max(self.batch_size, 1)))


class EEGDataLoader:
    @staticmethod
    def scale_data(data):
        if data is None:
            return None
        if data.ndim != 3:
            raise ValueError("Expected data with shape (samples, channels, time).")
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / (std + 1e-8)

    @staticmethod
    def to_dataloader(X, y, batch_size=64, num_workers=0, sample_type="random"):
        if X.ndim == 3:
            X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long).view(-1)
        data = TensorDataset(X_tensor, y_tensor)
        if sample_type == "balance":
            batch_sampler = BalancedBatchSampler(y_tensor.cpu().numpy(), batch_size=batch_size, shuffle=True)
        elif sample_type == "random":
            batch_sampler = BatchSampler(RandomSampler(data), batch_size=batch_size, drop_last=False)
        elif sample_type == "sequential":
            batch_sampler = BatchSampler(SequentialSampler(data), batch_size=batch_size, drop_last=False)
        else:
            raise ValueError("sample_type must be 'balance', 'random', or 'sequential'.")
        return DataLoader(data, num_workers=num_workers, batch_sampler=batch_sampler)
