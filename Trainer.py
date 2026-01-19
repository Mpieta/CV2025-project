import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models.video import r3d_18, R3D_18_Weights
from Dataset import ExerciseVideoDataModule
from sparse_model import SparseModel



class ExerciseDetector(pl.LightningModule):
    def __init__(self, num_classes=16, lr=1e-3, model_name = "pretrained_resnet"):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # pretrained resnet3d
        if model_name == "pretrained_resnet":
            self.model = r3d_18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "mobilenet_v2_3d":
            self.model = SparseModel(num_classes=num_classes)

        weights = torch.ones(num_classes)

        # less importance for class -1
        weights[0] = 0.5
        self.register_buffer("loss_weights", weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.loss_weights)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':

    dm = ExerciseVideoDataModule(
        video_dir="dataset/dataset/anon",
        label_dir="dataset/labels",
        split_csv="dataset/split.csv",
        batch_size=32
    )

    model = ExerciseDetector(num_classes=17, model_name="mobilenet_v2_3d")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=1,
    )

    trainer.fit(model, dm)
    torch.save(model.state_dict(), "exercise_model_final.pt")