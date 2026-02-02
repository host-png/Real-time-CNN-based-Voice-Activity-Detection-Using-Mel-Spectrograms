import pandas as pd
import sounddevice as sd
import soundfile as sf
import os
from melToPng import wav_to_mel_png

# esc50的人声过滤

df = pd.read_csv("E:\dataTrain\ESC-50-master\meta\esc50.csv")
filterVoice=df[(df['category'] != 'laughing') &
    (df['category'] != 'breathing') &
    (df['category'] != 'coughing') &
    (df['category'] != 'sneezing') &
    (df['category'] != 'snoring') &
    (df['category'] != 'crying_baby')]
pathEscWav = filterVoice['filename']
# 做非人声表
trainFrameData = pd.DataFrame({'Category':'unHuaman','path': "E:\\dataTrain\\ESC-50-master\\audio\\newWav\\" + pathEscWav
})

# 文件读取测试
# data, sr = sf.read(trainFrameData['path'][1])  # 读 wav
# sd.play(data, sr)  # 播放
# sd.wait()
hf = pd.read_csv("E:\\dataTrain\\cv-corpus-23.0-2025-09-05\\ja\\train.tsv", sep='\t')
#训练集本身就已经做过了过滤把投票数大于否票数的给筛选出来了
print(hf.columns)
wavFileBackSet= hf['path'].str.replace(".mp3", ".wav")
print(wavFileBackSet)
# 人声表
humanFrame = pd.DataFrame({'Category':'Huaman','path': "E:\\dataTrain\\cv-corpus-23.0-2025-09-05\\ja\\clips\\newWav\\" + wavFileBackSet
})
humanFrame = humanFrame.sample(n=len(trainFrameData), random_state=42)

# 可再进行的操作人声的样本数等于非人声的样本数
# 合并 数据集处理ok、
# 二元分类表
trainFrameData = pd.concat([trainFrameData, humanFrame], axis=0, ignore_index=True)
print(trainFrameData)
# 文件读取测试
# data, sr = sf.read(trainFrameData['path'][3518])  # 读 wav
# sd.play(data, sr)  # 播放
# sd.wait()


# wav_file = r"E:\AI\PythonProject1\1-137-A-32.wav"
# png_file = r"E:\AI\PythonProject1\1-137-A-32.png"
#
# for i in range(len(trainFrameData)):
#     wav_path = trainFrameData['path'][i]
#     filename = os.path.splitext(os.path.basename(wav_path))[0]
#     png_path = os.path.join(
#         r"E:\dataTrain\3000DataPngMel",
#         filename + ".png"
#     )
#     wav_to_mel_png(wav_path, png_path)

