import os
import cv2
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


RAW_LABELS = [-1, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LABEL_MAP = {raw: i for i, raw in enumerate(RAW_LABELS)}


class ExerciseVideoDataset(Dataset):
    def __init__(self, ids, video_dir, label_dir, clip_length=16, transform=None, neg_ratio=0.3):
        """
        Args:
            ids (list): List of video IDs (from split.csv)
            video_dir (str): Path to folder containing .mp4 files
            label_dir (str): Path to folder containing .csv files
            clip_length (int): Number of frames per sample (16 is standard)
            transform (callable): Albumentations or Torchvision transforms
            neg_ratio (float): Percentage of 'No Exercise' clips to keep (prevents bias)
        """
        self.ids = ids
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.clip_length = clip_length
        self.transform = transform
        self.neg_ratio = neg_ratio
        self.ram_cache = {}
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for vid_id in self.ids:
            video_path = os.path.join(self.video_dir, f"{vid_id}.mp4")
            label_path = os.path.join(self.label_dir, f"{vid_id}.csv")

            if not os.path.exists(video_path) or not os.path.exists(label_path):
                continue

            df = pd.read_csv(label_path, header=None)

            for start_idx in range(0, len(df) - self.clip_length, self.clip_length // 2):
                raw_label = df.iloc[start_idx + self.clip_length // 2, 2]
                if raw_label == -1 and random.random() > self.neg_ratio:
                    continue

                if raw_label in LABEL_MAP:
                    self.samples.append({
                        'video_path': video_path,
                        'start_frame': int(df.iloc[start_idx, 0]),
                        'label': LABEL_MAP[raw_label],
                        'cache_key': f"{vid_id}_{start_idx}"
                    })
        print(f"Total clips: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        key = sample['cache_key']

        if key in self.ram_cache:
            return self.ram_cache[key], sample['label']

        cap = cv2.VideoCapture(sample['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_frame'] - 1)

        clip = []
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame)
            if self.transform:
                img = self.transform(img)
            clip.append(img)

        cap.release()

        video_tensor = torch.stack(clip, dim=1)

        if len(self.ram_cache) < 1000:
            self.ram_cache[key] = video_tensor

        return video_tensor, sample['label']


import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ExerciseVideoDataModule(pl.LightningDataModule):
    def __init__(self, video_dir, label_dir, split_csv, batch_size=16, clip_length=16):
        super().__init__()
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.split_csv = split_csv
        self.batch_size = batch_size
        self.clip_length = clip_length

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4321, 0.3946, 0.3764],
                                 std=[0.2280, 0.2214, 0.2169])
        ])

    def setup(self, stage=None):
        df_split = pd.read_csv(self.split_csv)

        train_ids = df_split[df_split['split'] == 'train']['id'].tolist()
        val_ids = df_split[df_split['split'] == 'test']['id'].tolist()

        if stage == 'fit' or stage is None:
            self.train_ds = ExerciseVideoDataset(
                train_ids, self.video_dir, self.label_dir,
                clip_length=self.clip_length, transform=self.transform
            )
            self.val_ds = ExerciseVideoDataset(
                val_ids, self.video_dir, self.label_dir,
                clip_length=self.clip_length, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True)
