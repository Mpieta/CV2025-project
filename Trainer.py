import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models.video import r3d_18
from Dataset import ExerciseVideoDataModule
from sparse_model import SparseModel

class ExerciseDetector(pl.LightningModule):
    def __init__(self, num_classes=16, lr=1e-3, model_name="pretrained_resnet"):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        if model_name == "pretrained_resnet":
            self.model = r3d_18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "mobilenet_v2_3d":
            self.model = SparseModel(num_classes=num_classes)

        weights = torch.ones(num_classes)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode to save augmented videos')
    parser.add_argument('--model', type=str, default="mobilenet_v2_3d", choices=["mobilenet_v2_3d", "pretrained_resnet"])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    dm = ExerciseVideoDataModule(
        video_dir="dataset/dataset/anon",
        label_dir="dataset/labels",
        split_csv="dataset/split.csv",
        batch_size=16, 
        debug=args.debug
    )

    model = ExerciseDetector(num_classes=17, model_name=args.model)

    limit_batches = 1.0
    if args.debug:
        print("!!! DEBUG MODE ACTIVE - Training will act as sanity check !!!")
        limit_batches = 0.05 
        
    trainer = pl.Trainer(
        accelerator="auto", #zamienić na "gpu" jeśli dostępne
        devices=1,
        precision="16-mixed",
        max_epochs=args.epochs,
        limit_train_batches=limit_batches if args.debug else 1.0 
    )

    trainer.fit(model, dm)
    
    if not args.debug:
        torch.save(model.state_dict(), f"exercise_model_{args.model}_final.pt")