import os
import shutil

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from mP3Rms0ToWav import mp3_to_wav_rms
# =============================
# 1️构建数据集表
# =============================

def copy_audio_files(
    src_paths,
    dst_dir: str,
    desc: str = "Copying audio"
):
    """
    批量拷贝音频文件到指定目录

    参数:
        src_paths: 可迭代对象（如 list / pandas.Series），元素为源文件路径
        dst_dir: 目标目录
        desc: tqdm 进度条描述
    """

    os.makedirs(dst_dir, exist_ok=True)

    missing_files = []
    copied_count = 0

    for src_path in tqdm(src_paths, desc=desc):
        src_path = str(src_path)

        if not os.path.exists(src_path):
            missing_files.append(src_path)
            continue

        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)

        # copy2：保留时间戳等元信息
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    print(f"\n✅ 拷贝完成，共复制 {copied_count} 个文件")
    if missing_files:
        print(f"⚠️ 有 {len(missing_files)} 个文件不存在（已跳过）")

    return copied_count, missing_files


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

wav_paths = hf['path']

human_df = pd.DataFrame({
    'Category': 'Huaman',
    'path': "E:\\dataTrain\\cv-corpus-23.0-2025-09-05\\ja\\clips\\" + wav_paths
})

# 人声与非人声数量对齐
human_df = human_df.sample(n=int(len(unhuman_df) * 1.5), random_state=42)



dst_unhuman = r"E:\dataTrain\NewHUmanAndUnhuman"

copy_audio_files(
    src_paths=unhuman_df['path'],
    dst_dir=dst_unhuman,
    desc="Copying UNHUMAN audio"
)


dst_human = r"E:\dataTrain\NewHUmanAndUnhuman\humanMp3"

copy_audio_files(
    src_paths=human_df['path'],
    dst_dir=dst_human,
    desc="Copying HUMAN audio"
)

# 人声静音过滤
# input_dir = r"E:\dataTrain\NewHUmanAndUnhuman\humanMp3"
# output_dir = r"E:\dataTrain\NewHUmanAndUnhuman"
#
# mp3_to_wav_rms(input_dir, output_dir)
#

# 复制一份 DataFrame 避免修改原始
human_df_copy = human_df.copy()
unhuman_df_copy = unhuman_df.copy()

# ----------------- 人声改后缀为 .wav -----------------
human_df_copy['path'] = human_df_copy['path'].apply(
    lambda x: os.path.splitext(os.path.basename(x))[0] + ".wav"
)

# ----------------- 非人声保持原始文件名 -----------------
unhuman_df_copy['path'] = unhuman_df_copy['path'].apply(
    lambda x: os.path.basename(x)
)

# ----------------- 合并 -----------------
all_df = pd.concat([human_df_copy, unhuman_df_copy], ignore_index=True)

# ----------------- 添加统一目录 -----------------
base_dir = r"E:\dataTrain\NewHUmanAndUnhuman"
all_df['path'] = all_df['path'].apply(lambda x: os.path.join(base_dir, x))

# ----------------- 打乱顺序 -----------------
all_df = all_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# ----------------- 保存 CSV -----------------
csv_path = os.path.join(base_dir, "dataset.csv")
all_df.to_csv(csv_path, index=False)

print(f"✅ CSV 已生成，共 {len(all_df)} 条记录 -> {csv_path}")