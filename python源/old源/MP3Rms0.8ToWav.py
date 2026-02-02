import os
import numpy as np
from pydub import AudioSegment
import wave


# ================= RMS 相关 =================
def calc_rms(frame: np.ndarray) -> float:
    frame = frame.astype(np.float32)
    return np.sqrt(np.mean(frame * frame))


def remove_silence(pcm: np.ndarray, sr: int):
    """
    pcm: int16 numpy array (mono)
    sr: sample rate
    """
    frame_len = sr // 20      # 50 ms
    frame_shift = sr // 40    # 25 ms

    rms_list = []
    for i in range(0, len(pcm) - frame_len + 1, frame_shift):
        rms_list.append(calc_rms(pcm[i:i + frame_len]))

    if not rms_list:
        return pcm

    avg_rms = np.mean(rms_list)
    rms_threshold = avg_rms * 0.8

    print(f"Avg RMS={avg_rms:.2f}, Threshold={rms_threshold:.2f}")

    out = []
    last_frame_end = 0

    for i in range(0, len(pcm) - frame_len + 1, frame_shift):
        rms = calc_rms(pcm[i:i + frame_len])

        if rms > rms_threshold:
            copy_start = max(i, last_frame_end)
            copy_end = i + frame_len
            if copy_end > copy_start:
                out.append(pcm[copy_start:copy_end])
                last_frame_end = copy_end

    if not out:
        return np.array([], dtype=np.int16)

    return np.concatenate(out).astype(np.int16)


# ================= WAV 写出 =================
def write_wav_16bit(path, pcm, sr):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ================= MP3 -> RMS =================
def mp3_to_wav_rms(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".mp3"):
            continue

        mp3_path = os.path.join(input_dir, filename)
        out_name = os.path.splitext(filename)[0] + ".wav"
        out_path = os.path.join(output_dir, out_name)

        # 1️⃣ 读取 MP3
        audio = AudioSegment.from_mp3(mp3_path)

        # 2️⃣ 转成 16kHz / 16bit / mono
        audio = (
            audio
            .set_frame_rate(16000)
            .set_sample_width(2)
            .set_channels(1)
        )

        # 3️⃣ 转 numpy int16
        pcm = np.array(audio.get_array_of_samples(), dtype=np.int16)

        # 4️⃣ RMS 去静音
        pcm_clean = remove_silence(pcm, 16000)

        # 5️⃣ 写 WAV
        write_wav_16bit(out_path, pcm_clean, 16000)

        print(f"Processed: {mp3_path} -> {out_path}")


if __name__ == "__main__":
    input_dir = r"E:\dataTrain\ceshi"
    output_dir = r"E:\dataTrain\ceshi\okWav"

    mp3_to_wav_rms(input_dir, output_dir)
