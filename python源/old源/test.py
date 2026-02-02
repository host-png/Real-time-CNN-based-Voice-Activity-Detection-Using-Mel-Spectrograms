import wave
import numpy as np
import os
import pandas as pd
import sounddevice as sd
import soundfile as sf

df = pd.read_csv("E:\dataTrain\ESC-50-master\meta\esc50.csv")
import sounddevice as sd
import soundfile as sf

# print("列名:")
# print(df.columns)
# # 2️⃣ 查看前 5 行
# print("前5行数据:")
# print(df.head())
# df = pd.read_csv("esc50.csv")
# num_categories = df['category'].nunique()  # 计算不同类别的数量
# print("总类别数:", num_categories)
# var = df[df['target'] == 0]
# print(var)
# print(df['category'].unique())
# wavFileArry = df[df['category'] == 'insects']
# servui = wavFileArry['filename']
# print(len(servui))
# for i in range(5):
#     data, sr = sf.read(r"E:\dataTrain\ESC-50-master\audio\newWav" + "\\" + servui.iloc[i])  # 读 wav
#     sd.play(data, sr)  # 播放
#     sd.wait()
#

filterVoice=df[(df['category'] != 'laughing') &
    (df['category'] != 'breathing') &
    (df['category'] != 'coughing') &
    (df['category'] != 'sneezing') &
    (df['category'] != 'snoring') &
    (df['category'] != 'crying_baby')]
print(len(filterVoice))
# 等待播放结束


#print(df[df['category'] == 'laughing'])

# laughing → 笑声
#
# breathing → 呼吸声
#
# coughing → 咳嗽声
#
# sneezing → 打喷嚏
#
# snoring → 打鼾
#
# crying_baby → 婴儿哭声
