import os
from pydub import AudioSegment


def mp3_to_wav_16khz_16bit(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".mp3"):
            continue

        mp3_path = os.path.join(input_dir, filename)
        wav_name = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_dir, wav_name)

        try:
            audio = AudioSegment.from_mp3(mp3_path)

            # 16kHz / 16-bit / mono
            audio = (
                audio
                .set_frame_rate(16000)
                .set_sample_width(2)  # 16-bit
                .set_channels(1)
            )

            audio.export(wav_path, format="wav")
            print(f"[OK] {filename}")

        except Exception as e:
            print(f"[FAIL] {filename}: {e}")


if __name__ == "__main__":
    input_dir = r"E:\dataTrain\ceshi"
    output_dir = r"E:\dataTrain\ceshi\wav"

    mp3_to_wav_16khz_16bit(input_dir, output_dir)
