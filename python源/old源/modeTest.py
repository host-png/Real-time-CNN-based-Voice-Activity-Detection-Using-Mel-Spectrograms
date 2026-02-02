import pandas as pd
import path
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CNNnetWork import CnnNetVoice
import sys

from CNNnetWork import CnnNetVoice
import torch

def load_npy_mel(npy_path):
    """
    输入: npy 文件路径
    输出: [1, n_mels, time] 的 torch.Tensor
    """
    mel = np.load(npy_path)  # 直接是 ndarray
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # 添加 channel
    return mel_tensor


allData = pd.read_csv(r"E:\dataTrain\npyData50ms2weight\50ms2WeightMel.csv")
train_data, test_data = train_test_split(
    allData,
    test_size=0.3,    # 30% 作为测试集
    random_state=42,   # 保证可复现
    shuffle=True
)

print("测试训练分割ok")


device = torch.device("cpu")

model = CnnNetVoice()
model.load_state_dict(torch.load("cnnNetVoice_8.pth", map_location=device))
model.to(device)
model.eval()  # 非常重要！
error_records = []#存放错误样本
for i in range(len(test_data)):
    with torch.no_grad():
        inputTensor = load_npy_mel(test_data['path'].iloc[i])
        target = int(torch.tensor([[test_data['label'].iloc[i]]], dtype=torch.float32))
        output = model(inputTensor)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).item()
        # print("概率:", prob.item())
        if(pred != target):
            print(test_data['path'].iloc[i] + "当前i为" +str(i))
            error_records.append({
                "path":  test_data['path'].iloc[i],
            })



print("错误率" +str(len(error_records)/len(test_data)*100))
common = 0
for record in error_records:
    if "common" in record["path"]:
        common += 1


print("common占比" + str(100*common/len(error_records)))

#
# # 二分类损失函数 自动sigmoid
# lossFu = nn.BCEWithLogitsLoss()
# cnnNetVoice = CnnNetVoice()
# # 反向传播优化调参 随机梯度下降
# optimizer = torch.optim.SGD(cnnNetVoice.parameters(), lr=0.001)
# # 测试训练的次数
# trainStep =0
# testStep =0
# # Tensorbaord
# writer = SummaryWriter("./log")
#
# for i in range(10):
#     print("第"+str(i+1)+"次训练集训练")
#     cnnNetVoice.train()
#     print("总训练次数" + str(trainStep))
#     for num in range(len(train_data)):
#         inputTensor = load_npy_mel(train_data['path'].iloc[num])
#         target = torch.tensor([[train_data['label'].iloc[num]]], dtype=torch.float32)
#         outPut = cnnNetVoice(inputTensor)
#         loss = lossFu(outPut, target)
#         optimizer.zero_grad()
#         loss.backward()#反向传播
#         optimizer.step()
#         trainStep += 1
#         #  打印实时进度
#         progress = (num + 1) / len(train_data) * 100
#         sys.stdout.write(f"\r本轮训练进度: {num + 1}/{len(train_data)} ({progress:.2f}%)  当前 loss: {loss.item():.4f}")
#         sys.stdout.flush()  # 刷新输出缓冲区
#         if(trainStep % 100 == 0):#100次记录一次
#             writer.add_scalar("100尺度下训练损失随次数变换", loss.item(), trainStep)#写入tensor
#         if(trainStep % 1000 == 0):
#             writer.add_scalar("1000尺度下训练损失随次数变换", loss.item(), trainStep)  # 写入tensor
#
#     with torch.no_grad():#测试集验证
#         for num in range(len(test_data)):
#             inputTensor = load_npy_mel(test_data['path'].iloc[num])
#             target = torch.tensor([[test_data['label'].iloc[num]]], dtype=torch.float32) #真实值
#             outPut = cnnNetVoice(inputTensor)#预测值
#             loss = lossFu(outPut, target)
#             testStep += 1
#             progress = (num + 1) / len(test_data) * 100
#             sys.stdout.write(f"\r本轮测试: {num + 1}/{len(test_data)} ({progress:.2f}%)  当前 loss: {loss.item():.4f}")
#             sys.stdout.flush()  # 刷新输出缓冲区
#             if(testStep % 100 == 0):
#                 writer.add_scalar("100尺度测试损失随次数变化", loss.item(), testStep)
#             if(testStep % 1000 == 0):
#                 writer.add_scalar("1000尺度测试损失随次数变化", loss.item(), testStep)
#
# writer.close()
