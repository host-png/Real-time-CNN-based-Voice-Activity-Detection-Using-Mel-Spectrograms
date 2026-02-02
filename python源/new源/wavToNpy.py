import os
import soundfile as sf
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

df = pd.read_csv(r"E:\dataTrain\NewHUmanAndUnhuman\dataset.csv")

# 标签映射
label_map = {'Huaman': 1, 'unHuaman': 0}
df['label'] = df['Category'].map(label_map)

print(f"📊 原始 wav 文件数: {len(df)}")

sample_rate = 16000

slice_ms = 50        # 每个样本 = 50ms（模型输入）
n_fft_ms = 25        # STFT 窗口 = 25ms
hop_ms = 25          # STFT 步长 = 25ms
n_mels = 50          # Mel 频率维度

n_fft = int(sample_rate * n_fft_ms / 1000)       # 400
hop_length = int(sample_rate * hop_ms / 1000)    # 400
slice_samples = int(sample_rate * slice_ms / 1000)  # 800

output_dir = r"E:\dataTrain\NewHUmanAndUnhuman\npyData50ms2weight"
os.makedirs(output_dir, exist_ok=True)

# =============================
# 切片 + Mel → npy
# =============================

def slice_wav_to_npy(wav_path, label):
    records = []

    try:
        waveform, sr = sf.read(wav_path)  # waveform: [num_samples] 或 [num_samples, channels]

        # 转 torch.Tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # 单声道
        if waveform.ndim > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(1)  # shape -> [num_samples, 1]

        waveform = waveform.T  # 转为 [1, L]，保持和 torchaudio 一致

        # 重采样
        if sr != sample_rate:
            L = waveform.shape[1]
            new_len = int(L * sample_rate / sr)
            waveform = torch.tensor(resample(waveform.numpy(), new_len, axis=1), dtype=torch.float32)

        L = waveform.shape[1]

        mel_transform = torch.nn.Sequential(
            torch.nn.Conv1d(1, 1, 1)  # 占位，用于兼容结构，可忽略
        )
        # ⚠️ 这里使用 torchaudio.transforms.MelSpectrogram
        # 仍可使用 torchaudio 进行 Mel 变换：
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False
        )

        fname = os.path.splitext(os.path.basename(wav_path))[0]
        slice_count = 0

        for start in range(0, L - slice_samples + 1, slice_samples):
            segment = waveform[:, start:start + slice_samples]
            mel = mel_transform(segment)
            mel_db = 10 * torch.log10(mel + 1e-9)

            out_path = os.path.join(
                output_dir,
                f"{fname}_slice[{start}].npy"
            )

            np.save(out_path, mel_db.numpy())

            records.append({
                "path": out_path,
                "label": label
            })

            slice_count += 1

        print(f"[OK] {fname} → {slice_count} slices")
        return records

    except Exception as e:
        print(f"[ERROR] {wav_path}")
        print(e)
        return []

# =============================
# 遍历所有 wav
# =============================

all_records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="处理音频"):
    all_records.extend(
        slice_wav_to_npy(row['path'], row['label'])
    )

print(f"\n✅ 总切片数: {len(all_records)}")

# =============================
# 生成 CSV
# =============================

index_df = pd.DataFrame(all_records)
index_csv = os.path.join(output_dir, "50ms2WeightMel.csv")
index_df.to_csv(index_csv, index=False)

print(f"📄 CSV 已生成: {index_csv}")
