import pandas as pd
import sounddevice as sd
import soundfile as sf



df = pd.read_csv("E:\\dataTrain\\cv-corpus-23.0-2025-09-05\\ja\\train.tsv", sep='\t')
#训练集本身就已经做过了过滤把投票数大于否票数的给筛选出来了


# E:\dataTrain\cv-corpus-23.0-2025-09-05\ja\clips\newWav
# print("列名:")
print(df.columns)
# print(df[[ 'up_votes','down_votes']])
# upVoteFiler = df[(df['up_votes'] - df['down_votes'])>0]
# upVoteFiler1 = df[df['down_votes']>10]
# print(upVoteFiler1[['up_votes','down_votes']])
# print(df['path'])
wavFileBackSet= df['path'].str.replace(".mp3", ".wav")
print(wavFileBackSet)
humanFrame = pd.DataFrame({'Category':'Huaman','path': "E:\\dataTrain\\cv-corpus-23.0-2025-09-05\\ja\\clips\\newWav\\" + wavFileBackSet
})
print(humanFrame)
# data, sr = sf.read(humanFrame['path'][2])  # 读 wav
# sd.play(data, sr)  # 播放
# sd.wait()