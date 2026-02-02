import numpy as np

# npy 文件路径
npy_path = r"E:\dataTrain\npyData50ms2weight\1-137-A-32_slice[0].npy"
# 读取 npy
data = np.load(npy_path)

# 打印内容和形状
print("数据类型:", type(data))
print("数据形状:", data.shape)
print("前 10 个元素:", data.flatten()[:100])
