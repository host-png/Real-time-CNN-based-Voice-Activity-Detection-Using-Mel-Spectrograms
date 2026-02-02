import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import torch
import os


def wav_to_mel_png1(
        wav_path,
        png_path,
        sample_rate=16000,
        n_fft=800,
        hop_length=400,
        n_mels=50
):
    try:
        print(f"[INFO] 开始处理: {wav_path}")

        # 1️⃣ 用 soundfile 读 wav（不走 torchcodec）
        wav, sr = sf.read(wav_path)

        # 2️⃣ 单声道
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        waveform = torch.from_numpy(wav).float().unsqueeze(0)

        # 3️⃣ 重采样
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

        # 4️⃣ Mel
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel = mel_transform(waveform)

        # 5️⃣ log
        mel_db = 10 * torch.log10(mel + 1e-9)

        frames = mel_db.shape[-1]
        height = 3
        width = frames / 20

        # 6️⃣ 保存 PNG
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.figure(figsize=(width, height))
        plt.imshow(
            mel_db[0].numpy(),
            origin='lower',
            aspect='auto',
            cmap='magma'
        )
        plt.axis('off')
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"[SUCCESS] 已保存: {png_path}\n")

    except Exception as e:
        print(f"[ERROR] 处理失败: {wav_path}")
        print("原因:", e)
