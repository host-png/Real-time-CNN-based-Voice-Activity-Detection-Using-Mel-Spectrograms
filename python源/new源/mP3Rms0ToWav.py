import os
import numpy as np
from pydub import AudioSegment
import wave

# ============================================================
# RMS 计算相关
# ============================================================

def calc_rms(frame: np.ndarray) -> float:
    """
    计算一帧 PCM 数据的 RMS（均方根能量）

    参数:
        frame: numpy.ndarray，int16 或 float32 音频数据

    返回:
        float，RMS 值
    """
    frame = frame.astype(np.float32)
    return np.sqrt(np.mean(frame * frame))


def remove_silence(pcm: np.ndarray, sr: int) -> np.ndarray:
    """
    使用 RMS 阈值法进行简单的静音段裁剪

    参数:
        pcm: numpy.ndarray，int16，单声道 PCM 音频
        sr: 采样率（如 16000）

    返回:
        numpy.ndarray，去除静音后的 PCM（int16）
    """

    # 帧长度：50 ms
    frame_len = sr // 20

    # 帧移：25 ms
    frame_shift = sr // 40

    rms_list = []

    # 先遍历一遍，统计所有帧的 RMS
    for i in range(0, len(pcm) - frame_len + 1, frame_shift):
        frame = pcm[i:i + frame_len]
        rms_list.append(calc_rms(frame))

    # 如果音频太短，没有有效帧，直接返回原始 PCM
    if not rms_list:
        return pcm

    avg_rms = np.mean(rms_list)

    # RMS 阈值（经验值：平均 RMS 的 0.8）
    rms_threshold = avg_rms * 0.8

    print(f"[RMS] 平均值={avg_rms:.2f}, 阈值={rms_threshold:.2f}")

    out_frames = []
    last_frame_end = 0

    # 再遍历一遍，根据阈值决定是否保留
    for i in range(0, len(pcm) - frame_len + 1, frame_shift):
        frame = pcm[i:i + frame_len]
        rms = calc_rms(frame)

        if rms > rms_threshold:
            # 防止帧之间重复拷贝
            copy_start = max(i, last_frame_end)
            copy_end = i + frame_len

            if copy_end > copy_start:
                out_frames.append(pcm[copy_start:copy_end])
                last_frame_end = copy_end

    # 如果全部被判定为静音
    if not out_frames:
        return np.array([], dtype=np.int16)

    return np.concatenate(out_frames).astype(np.int16)


# ============================================================
# WAV 写文件（16-bit PCM）
# ============================================================

def write_wav_16bit(path: str, pcm: np.ndarray, sr: int):
    """
    将 int16 PCM 写成 16-bit 单声道 WAV 文件

    参数:
        path: 输出 wav 路径
        pcm: numpy.ndarray，int16 PCM 数据
        sr: 采样率
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)      # 单声道
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ============================================================
# MP3 → WAV + RMS 去静音 主流程
# ============================================================

def mp3_to_wav_rms(input_dir: str, output_dir: str, target_sr: int = 16000):
    """
    批量处理 MP3 文件：
    MP3 → 16kHz / 16-bit / mono → RMS 去静音 → WAV

    参数:
        input_dir: MP3 输入目录
        output_dir: WAV 输出目录
        target_sr: 目标采样率（默认 16000）
    """

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".mp3"):
            continue

        mp3_path = os.path.join(input_dir, filename)
        wav_name = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_dir, wav_name)

        # 1️⃣ 读取 MP3 文件
        audio = AudioSegment.from_mp3(mp3_path)

        # 2️⃣ 重采样 + 位深 + 单声道
        audio = (
            audio
            .set_frame_rate(target_sr)
            .set_sample_width(2)   # 16-bit
            .set_channels(1)
        )

        # 3️⃣ 转成 numpy int16 PCM
        pcm = np.array(audio.get_array_of_samples(), dtype=np.int16)

        # 4️⃣ RMS 去静音
        pcm_clean = remove_silence(pcm, target_sr)

        # 5️⃣ 写 WAV 文件
        write_wav_16bit(wav_path, pcm_clean, target_sr)

        print(f"[OK] {mp3_path} -> {wav_path}")


# ============================================================
# 示例（作为脚本运行时才会执行）
# ============================================================

if __name__ == "__main__":
    input_dir = r"E:\dataTrain\ceshi"
    output_dir = r"E:\dataTrain\ceshi\okWav"

    mp3_to_wav_rms(input_dir, output_dir)
