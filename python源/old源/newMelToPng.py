from pathlib import Path
import pandas as pd
import os
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from melToPng import wav_to_mel_png
from melToPngNewPack import wav_to_mel_png1

root = Path(r"E:\dataTrain\ceshi\okWav")

data = []

for wav in root.rglob("*.wav"):
    label = wav.parent.name
    data.append({
        "path": str(wav),
        "label": label
    })

df = pd.DataFrame(data)
print(len(df))
print(df.head())

out_dir = Path(r"E:\dataTrain\ceshi\okWav\Melpng")
out_dir.mkdir(parents=True, exist_ok=True)

for wav_path in df["path"]:
    filename = Path(wav_path).stem
    png_path = out_dir / f"{filename}.png"
    wav_to_mel_png1(wav_path, str(png_path))
