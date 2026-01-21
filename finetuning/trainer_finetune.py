import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparse_model import SparseModel
from torchvision.models.video import r3d_18


class FinetuneExerciseDetector(pl.LightningModule):
    def __init__(self, base_model_name="mobilenet_v2_3d", num_classes=16, lr=1e-4, consistency_weight=10.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.consistency_weight = consistency_weight

        if base_model_name == "mobilenet_v2_3d":
            self.model = SparseModel(num_classes=num_classes)
        else:
            self.model = r3d_18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        weights = torch.ones(num_classes)
        weights[0] = 0.5
        self.register_buffer("loss_weights", weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clip_a, clip_b, y = batch

        logits_a = self(clip_a)
        logits_b = self(clip_b)

        loss_cls = F.cross_entropy(logits_a, y, weight=self.loss_weights)

        log_probs_a = F.log_softmax(logits_a, dim=1)
        log_probs_b = F.log_softmax(logits_b, dim=1)

        loss_kl_ab = F.kl_div(log_probs_a, log_probs_b, log_target=True, reduction='batchmean')
        loss_kl_ba = F.kl_div(log_probs_b, log_probs_a, log_target=True, reduction='batchmean')

        loss_cons = (loss_kl_ab + loss_kl_ba) / 2

        total_loss = loss_cls + (self.consistency_weight * loss_cons)

        self.log('ft_loss', total_loss, prog_bar=True)
        self.log('ft_cls', loss_cls, prog_bar=True)
        self.log('ft_cons', loss_cons, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)