import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, recall_score

from EEGNet import EEGNetBackbone
from EEGTool import EEGDataLoader


def _orthogonal_projection_matrices(feat, label):
    unique_labels = torch.unique(label)
    class_means = []
    for cls in unique_labels:
        class_feat = feat[label == cls]
        class_means.append(class_feat.mean(dim=0))
    mu = torch.stack(class_means)
    mu_all = mu.mean(dim=0, keepdim=True)
    mu_diff = mu - mu_all
    sb = mu_diff.T @ mu_diff
    eigvals, eigvecs = torch.linalg.eigh(sb)
    eigvecs = eigvecs[:, torch.argsort(eigvals, descending=True)]
    rank = min(max(int(unique_labels.numel()) - 1, 1), feat.shape[1])
    basis = eigvecs[:, :rank]
    ps = basis @ basis.T
    pg = torch.eye(feat.shape[1], dtype=feat.dtype, device=feat.device) - ps
    return ps, pg


class ODRAProjectionDecomposer:
    def __init__(self):
        self.fitted_ = False
        self.specific_projection_ = None
        self.general_projection_ = None

    def fit(self, feat, label):
        feat = torch.as_tensor(feat)
        label = torch.as_tensor(label, device=feat.device)
        ps, pg = _orthogonal_projection_matrices(feat, label)
        self.specific_projection_ = ps
        self.general_projection_ = pg
        self.fitted_ = True
        return self

    def transform(self, feat):
        if not self.fitted_:
            raise RuntimeError("The decomposer must be fitted before calling transform().")
        feat = torch.as_tensor(feat)
        ps = self.specific_projection_.to(device=feat.device, dtype=feat.dtype)
        pg = self.general_projection_.to(device=feat.device, dtype=feat.dtype)
        return feat @ ps, feat @ pg


class ODRA(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        backbone=EEGNetBackbone,
        alpha=1.0,
        device=None,
        finetune_lr=2e-4,
    ):
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        self.alpha = float(alpha)
        self.num_channels = int(num_channels)
        self.num_classes = int(num_classes)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.backbone = backbone(num_classes=num_classes, num_channels=num_channels)
        self.fc = nn.Sequential(nn.LazyLinear(num_classes))
        self.loss_fn = nn.CrossEntropyLoss()
        self.finetune_lr = float(finetune_lr)
        self.decomposer = None
        self.class_dict = None
        self.nontarget = None
        self.target_classes = None
        self.train_loss = []
        self.training_time = None
        self.to(self.device)
        self.loss_fn.to(self.device)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat)

    def _sample_indices(self, num_items, sample_size):
        return torch.randint(0, num_items, (sample_size,), device=self.device)

    def build_class_dict(self, X_train_feat, y_train):
        feat = torch.as_tensor(X_train_feat, dtype=torch.float32, device=self.device)
        label = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self.decomposer = ODRAProjectionDecomposer().fit(feat, label)
        X_s, X_g = self.decomposer.transform(feat)
        class_dict = {}
        for cls in torch.unique(label).tolist():
            mask = label == int(cls)
            class_dict[int(cls)] = (X_s[mask], X_g[mask], feat[mask])
        class_dict["all"] = (X_s, X_g, feat)
        return class_dict

    def augment(self):
        all_feat = []
        all_label = []
        data_nontar = self.class_dict[self.nontarget]
        data_all = self.class_dict["all"]
        for cls in self.target_classes:
            data_cls = self.class_dict[cls]
            aug_num = max(int(data_nontar[0].shape[0] - data_cls[0].shape[0]), 0)
            if aug_num > 0:
                idx_tar = self._sample_indices(data_cls[0].shape[0], aug_num)
                idx_all = self._sample_indices(data_all[0].shape[0], aug_num)
                specific_feat = data_cls[0][idx_tar]
                general_feat = data_all[1][idx_all]
                low = 1.0 - self.alpha
                high = 1.0 + self.alpha
                lambda1 = (high - low) * torch.rand((aug_num, 1), device=self.device) + low
                lambda2 = (high - low) * torch.rand((aug_num, 1), device=self.device) + low
                aug_feat = lambda1 * specific_feat + lambda2 * general_feat
                all_feat.append(aug_feat)
                all_label.append(torch.full((aug_num,), int(cls), device=self.device))
            all_feat.append(data_cls[2])
            all_label.append(torch.full((data_cls[2].shape[0],), int(cls), device=self.device))
        all_feat.append(data_nontar[2])
        all_label.append(torch.full((data_nontar[2].shape[0],), int(self.nontarget), device=self.device))
        return torch.cat(all_feat, dim=0), torch.cat(all_label, dim=0)

    def _extract_backbone_features(self, X, batch_size):
        loader = EEGDataLoader.to_dataloader(X=X, y=np.zeros(len(X), dtype=np.int64), batch_size=batch_size, sample_type="sequential")
        features = []
        self.backbone.eval()
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device).float()
                features.append(self.backbone(batch_X).detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    def fit(
        self,
        X_train,
        y_train,
        epochs=80,
        pretrain_lr=1e-3,
        finetune_epochs=120,
        verbose=True,
        batch_size=64,
        save_path="",
    ):
        start_time = time.time()
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        if X_train.ndim != 3:
            raise ValueError("X_train must have shape (samples, channels, time).")
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        self.nontarget = int(unique_labels[np.argmax(label_counts)])
        self.target_classes = [int(cls) for cls in unique_labels.tolist() if int(cls) != self.nontarget]

        train_loader = EEGDataLoader.to_dataloader(
            X=X_train,
            y=y_train,
            batch_size=batch_size,
            sample_type="balance",
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=pretrain_lr)
        self.train_loss = []
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device).long()
                optimizer.zero_grad()
                logits = self(batch_X)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            avg_loss = total_loss / max(len(train_loader), 1)
            self.train_loss.append(avg_loss)
            if verbose:
                print(f"Pretrain Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.6f}")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.fc.train()
        finetune_optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.finetune_lr)

        X_train_feat = self._extract_backbone_features(X_train, batch_size=batch_size)
        self.class_dict = self.build_class_dict(X_train_feat, y_train)
        finetune_feat, finetune_label = self.augment()
        finetune_loader = EEGDataLoader.to_dataloader(
            X=finetune_feat.detach().cpu().numpy(),
            y=finetune_label.detach().cpu().numpy(),
            batch_size=batch_size,
            sample_type="random",
        )

        for epoch in range(1, finetune_epochs + 1):
            self.backbone.eval()
            self.fc.train()
            total_loss = 0.0
            for feat, label in finetune_loader:
                feat = feat.to(self.device).float()
                label = label.to(self.device).long()
                finetune_optimizer.zero_grad()
                logits = self.fc(feat)
                loss = self.loss_fn(logits, label)
                loss.backward()
                finetune_optimizer.step()
                total_loss += float(loss.item())
            avg_loss = total_loss / max(len(finetune_loader), 1)
            self.train_loss.append(avg_loss)
            if verbose:
                print(f"Finetune Epoch [{epoch}/{finetune_epochs}] - Loss: {avg_loss:.6f}")

        if save_path:
            self.save_model(save_path)

        self.training_time = time.time() - start_time
        return {
            "training_time": self.training_time,
            "training_loss": tuple(self.train_loss),
        }

    def predict(self, X, return_logits=True, batch_size=64):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, channels, time).")
        if X.shape[0] == 0:
            return np.array([]), np.array([])
        loader = EEGDataLoader.to_dataloader(
            X=X,
            y=np.zeros(X.shape[0], dtype=np.int64),
            batch_size=batch_size,
            sample_type="sequential",
        )
        self.eval()
        all_logits = []
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device).float()
                logits = self(batch_X)
                all_logits.append(logits.detach().cpu().numpy())
                all_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        return (all_preds, all_logits) if return_logits else all_preds

    def score(self, X, y_true, batch_size=64):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred, y_score = self.predict(X, return_logits=True, batch_size=batch_size)
        scores = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall-per-class": recall_score(y_true, y_pred, average=None, zero_division=0),
        }
        return scores

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_channels": self.num_channels,
                "num_classes": self.num_classes,
                "alpha": self.alpha,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint
