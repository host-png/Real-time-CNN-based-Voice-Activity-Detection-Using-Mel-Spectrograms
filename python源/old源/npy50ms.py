import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================
# 1️⃣ 构建数据集表
# =============================

# ESC-50 非人声
esc_df = pd.read_csv(r"E:\dataTrain\ESC-50-master\meta\esc50.csv")
filterVoice = esc_df[
    (esc_df['category'] != 'laughing') &
    (esc_df['category'] != 'breathing') &
    (esc_df['category'] != 'coughing') &
    (esc_df['category'] != 'sneezing') &
    (esc_df['category'] != 'snoring') &
    (esc_df['category'] != 'crying_baby')
]

unhuman_df = pd.DataFrame({
    'Category': 'unHuaman',
    'path': "E:\\dataTrain\\ESC-50-master\\audio\\newWav\\" + filterVoice['filename']
})

# Common Voice 人声
hf = pd.read_csv(
    r"E:\dataTrain\cv-corpus-23.0-2025-09-05\ja\train.tsv",
    sep='\t'
)

wav_paths = hf['path'].str.replace(".mp3", ".wav", regex=False)

human_df = pd.DataFrame({
    'Category': 'Huaman',
    'path': "E:\\dataTrain\\cv-corpus-23.0-2025-09-05\\ja\\clips\\newWav\\" + wav_paths
})

# 人声与非人声数量对齐
human_df = human_df.sample(n=len(unhuman_df), random_state=42)

# 合并
df = pd.concat([unhuman_df, human_df], ignore_index=True)

# 标签映射
label_map = {'Huaman': 1, 'unHuaman': 0}
df['label'] = df['Category'].map(label_map)

print(f"📊 原始 wav 文件数: {len(df)}")

# =============================
# 2️⃣ 参数设置
# =============================

sample_rate = 16000

slice_ms = 50        # 每个样本 = 50ms（模型输入）
n_fft_ms = 25        # STFT 窗口 = 25ms
hop_ms = 25          # STFT 步长 = 25ms
n_mels = 50          # Mel 频率维度

n_fft = int(sample_rate * n_fft_ms / 1000)       # 400
hop_length = int(sample_rate * hop_ms / 1000)    # 400
slice_samples = int(sample_rate * slice_ms / 1000)  # 800

output_dir = r"E:\dataTrain\npyData50ms2weight"
os.makedirs(output_dir, exist_ok=True)

# =============================
# 3️⃣ 切片 + Mel → npy
#    ✅ 同时返回 path + label
# =============================

def slice_wav_to_npy(wav_path, label):
    records = []

    try:
        waveform, sr = torchaudio.load(wav_path)

        # 单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重采样
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

        L = waveform.shape[1]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center = False # 防止补帧
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

            # ⚠️ 只保存 mel（label 写进 CSV）
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
# 4️⃣ 遍历所有 wav
# =============================

all_records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="处理音频"):
    all_records.extend(
        slice_wav_to_npy(row['path'], row['label'])
    )

print(f"\n✅ 总切片数: {len(all_records)}")

# =============================
# 5️⃣ 生成 CSV（⚡ 极快）
# =============================

index_df = pd.DataFrame(all_records)
index_csv = os.path.join(output_dir, "50ms2WeightMel.csv")
index_df.to_csv(index_csv, index=False)

print(f"📄 CSV 已生成: {index_csv}")
