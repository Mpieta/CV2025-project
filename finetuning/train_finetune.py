import pytorch_lightning as pl
import torch
from dataset_finetune import ConsistencyDataModule
from trainer_finetune import FinetuneExerciseDetector

CHECKPOINT_PATH = "lightning_logs/version_22/checkpoints/epoch=0-step=2816.ckpt"

def run_finetuning():
    dm = ConsistencyDataModule(
        video_dir="dataset/dataset/anon",
        label_dir="dataset/labels",
        split_csv="dataset/split.csv",
        batch_size=12,
        clip_length=16
    )

    model = FinetuneExerciseDetector(
        base_model_name="mobilenet_v2_3d",
        num_classes=17,
        lr=1e-5,
        consistency_weight=5.0
    )

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH)
        model.model.load_state_dict(state_dict)
    except:
        checkpoint = torch.load(CHECKPOINT_PATH)
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
        model.model.load_state_dict(state_dict, strict=False)

    print("Weights loaded successfully!")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=3,
        log_every_n_steps=10
    )

    trainer.fit(model, dm)

    torch.save(model.model.state_dict(), "../exercise_model_finetuned.pt")
    print("Finetuning complete! Model saved.")


if __name__ == "__main__":
    run_finetuning()