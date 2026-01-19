import os
import cv2
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2  
import pytorch_lightning as pl

RAW_LABELS = [-1, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LABEL_MAP = {raw: i for i, raw in enumerate(RAW_LABELS)}

class ExerciseVideoDataset(Dataset):
    def __init__(self, ids, video_dir, label_dir, clip_length=16, transform=None, neg_ratio=0.3, debug=False, debug_dir="debug_output"):
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
        
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[DEBUG MODE] Augmented videos will be saved to: {self.debug_dir}")

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
                        'cache_key': f"{vid_id}_{start_idx}",
                        'vid_id': vid_id 
                    })
        print(f"Total clips: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _save_debug_video(self, tensor, name_suffix):
        """
        Zapisuje wideo używając CV2 (bardziej niezawodne na Windows).
        Oczekuje tensora wejściowego w formacie: [Czas, Kanały, Wysokość, Szerokość]
        """
        try:
            vid = tensor.clone().detach().cpu()
            
            if vid.max() > 1.0 or vid.min() < 0.0:
                vid = vid - vid.min()
                vid = vid / (vid.max() + 1e-6)
            
            vid = (vid * 255).byte()
            
            vid = vid.permute(0, 2, 3, 1).numpy()
            
            T, H, W, C = vid.shape
            output_path = os.path.join(self.debug_dir, f"{name_suffix}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 10, (W, H))
            
            if not out.isOpened():
                print(f"Failed to open video writer for {output_path}")
                return

            for i in range(T):
                frame = vid[i] 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                out.write(frame)
            out.release()
            
        except Exception as e:
            print(f"Warning: Could not save debug video {name_suffix}: {e}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        key = sample['cache_key']

        if key in self.ram_cache and not self.debug:
            return self.ram_cache[key], sample['label']

        cap = cv2.VideoCapture(sample['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_frame'] - 1)

        frames_list = []
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(torch.from_numpy(frame))

        cap.release()

        video_tensor = torch.stack(frames_list)
        
        video_tensor = video_tensor.permute(0, 3, 1, 2) 

        should_save = False
        if self.debug:
            try:
                files_count = len([name for name in os.listdir(self.debug_dir) if name.endswith('.mp4')])
                if files_count < 10: 
                    should_save = True
            except:
                pass 

        if should_save: 
            vid_name = sample['vid_id'].replace('/', '_') 
            self._save_debug_video(video_tensor, f"vid_{vid_name}_idx{idx}_orig")

        if self.transform:
            video_tensor = self.transform(video_tensor)

        if should_save:
            vid_name = sample['vid_id'].replace('/', '_')
            self._save_debug_video(video_tensor, f"vid_{vid_name}_idx{idx}_aug")
            print(f"Saved debug videos for sample {idx}")

        video_tensor = video_tensor.permute(1, 0, 2, 3) 

        if len(self.ram_cache) < 1000 and not self.debug:
            self.ram_cache[key] = video_tensor 

        return video_tensor, sample['label']


class ExerciseVideoDataModule(pl.LightningDataModule):
    def __init__(self, video_dir, label_dir, split_csv, batch_size=16, clip_length=16, debug=False):
        super().__init__()
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.split_csv = split_csv
        self.batch_size = batch_size
        self.clip_length = clip_length
        self.debug = debug

        self.train_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(450),  
            v2.RandomResizedCrop(size=(400, 300), scale=(0.7, 1.0)), 
            v2.RandomHorizontalFlip(p=0.5), 
            v2.RandomRotation(degrees=10), 
            v2.Normalize(mean=[0.4321, 0.3946, 0.3764], std=[0.2280, 0.2214, 0.2169])
        ])

        self.val_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(450),
            v2.CenterCrop(size=(400, 300)),
            v2.Normalize(mean=[0.4321, 0.3946, 0.3764], std=[0.2280, 0.2214, 0.2169])
        ])

    def setup(self, stage=None):
        df_split = pd.read_csv(self.split_csv)
        train_ids = df_split[df_split['split'] == 'train']['id'].tolist()
        val_ids = df_split[df_split['split'] == 'test']['id'].tolist()

        if stage == 'fit' or stage is None:
            self.train_ds = ExerciseVideoDataset(
                train_ids, self.video_dir, self.label_dir,
                clip_length=self.clip_length, transform=self.train_transform,
                debug=self.debug 
            )
            self.val_ds = ExerciseVideoDataset(
                val_ids, self.video_dir, self.label_dir,
                clip_length=self.clip_length, transform=self.val_transform,
                debug=False 
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0, pin_memory=False)