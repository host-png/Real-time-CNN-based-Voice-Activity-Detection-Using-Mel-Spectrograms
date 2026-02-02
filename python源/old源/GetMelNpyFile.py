import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MelDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        npy_path = self.df.iloc[idx]['path']
        data = np.load(npy_path, allow_pickle=True).item()
        mel = data['mel']          # shape: [n_mels, time] 例如 [50, 2]
        label = data['label']      # 0 或 1

        # 转 torch tensor
        mel_tensor = torch.tensor(mel, dtype=torch.float32)

        # CNN 输入需要 [C,H,W]，加一个 channel 维度
        mel_tensor = mel_tensor.unsqueeze(0)  # [1, n_mels, time]

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return mel_tensor, label_tensor

# -----------------------------
# DataLoader
# -----------------------------
# dataset = MelDataset(r"E:\dataTrain\npyData50ms2weight\50ms2WeightMel.csv")
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
