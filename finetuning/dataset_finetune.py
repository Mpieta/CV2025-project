import os
import cv2
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

# Mapowania (takie same jak w projekcie)
RAW_LABELS = [-1, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LABEL_MAP = {raw: i for i, raw in enumerate(RAW_LABELS)}


class ConsistencyDataset(Dataset):
    def __init__(self, ids, video_dir, label_dir, clip_length=16, transform=None):
        self.ids = ids
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.clip_length = clip_length
        self.transform = transform
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        # Tutaj logika podobna jak w oryginale, ale bierzemy wszystko (bez pomijania negatywów)
        # albo z pomijaniem - zależnie jak chcesz. Tu wersja uproszczona:
        for vid_id in self.ids:
            video_path = os.path.join(self.video_dir, f"{vid_id}.mp4")
            label_path = os.path.join(self.label_dir, f"{vid_id}.csv")

            if not os.path.exists(video_path) or not os.path.exists(label_path):
                continue

            df = pd.read_csv(label_path, header=None)

            # Krok co clip_length (bez nakładania, żeby było szybciej dla finetuningu)
            for start_idx in range(0, len(df) - self.clip_length - 1, self.clip_length):
                raw_label = df.iloc[start_idx + self.clip_length // 2, 2]

                if raw_label in LABEL_MAP:
                    self.samples.append({
                        'video_path': video_path,
                        'start_frame': int(df.iloc[start_idx, 0]),
                        'label': LABEL_MAP[raw_label]
                    })
        print(f"Finetuning clips found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_frame'] - 1)

        frames = []
        # Pobieramy clip_length + 1 klatek (np. 17)
        for _ in range(self.clip_length + 1):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame)
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        cap.release()

        # Budujemy dwa tensory przesunięte o 1 klatkę
        # Clip A: klatki 0..15
        clip_a = torch.stack(frames[:-1], dim=1)
        # Clip B: klatki 1..16
        clip_b = torch.stack(frames[1:], dim=1)

        return clip_a, clip_b, sample['label']


class ConsistencyDataModule(pl.LightningDataModule):
    def __init__(self, video_dir, label_dir, split_csv, batch_size=8, clip_length=16):
        super().__init__()
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.split_csv = split_csv
        self.batch_size = batch_size  # Mniejszy batch bo mamy 2x więcej danych (pary)
        self.clip_length = clip_length

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4321, 0.3946, 0.3764], std=[0.2280, 0.2214, 0.2169])
        ])

    def setup(self, stage=None):
        df = pd.read_csv(self.split_csv)
        # Do finetuningu używamy tylko zbioru treningowego
        train_ids = df[df['split'] == 'train']['id'].tolist()

        self.train_ds = ConsistencyDataset(
            train_ids, self.video_dir, self.label_dir,
            clip_length=self.clip_length, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)