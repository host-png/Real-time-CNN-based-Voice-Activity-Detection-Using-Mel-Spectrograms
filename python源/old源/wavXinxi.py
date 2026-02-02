# import wave
# import numpy as np
# import os
#
# def get_wav_info(file_path):
#     with wave.open(file_path, 'rb') as wf:
#         n_channels = wf.getnchannels()       # 声道数
#         sample_width = wf.getsampwidth()     # 每个采样点字节数
#         frame_rate = wf.getframerate()       # 采样率
#         n_frames = wf.getnframes()           # 总帧数
#         duration = n_frames / frame_rate     # 时长（秒）
#
#         # 读取全部 PCM 数据
#         raw_data = wf.readframes(n_frames)
#         data = np.frombuffer(raw_data, dtype=np.int16)
#         if n_channels > 1:
#             data = data.reshape(-1, n_channels)
#
#     info = {
#         "channels": n_channels,
#         "sample_width_bytes": sample_width,
#         "sample_rate": frame_rate,
#         "n_frames": n_frames,
#         "duration_sec": duration,
#         "data_shape": data.shape,
#         "dtype": data.dtype
#     }
#     return info
#
# # 测试
# file_path = "common_voice_ja_19482480.wav"
# info = get_wav_info(file_path)
# for k, v in info.items():
#     print(f"{k}: {v}")
