import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CNNnetWork import CnnNetVoice
import sys

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

# 二分类损失函数 自动sigmoid
lossFu = nn.BCEWithLogitsLoss()
lossFu = lossFu.cuda()
cnnNetVoice = CnnNetVoice()
cnnNetVoice = cnnNetVoice.cuda()
# 反向传播优化调参 随机梯度下降
optimizer = torch.optim.SGD(cnnNetVoice.parameters(), lr=0.000001)

# 测试训练的次数
trainStep =0
testStep =0
# Tensorbaord
writer = SummaryWriter("./log")

for i in range(50):
    correct_total = 0;#准确率
    print("第"+str(i+1)+"次训练集训练")
    cnnNetVoice.train()
    print("总训练次数" + str(trainStep))
    for num in range(len(train_data)):
        inputTensor = load_npy_mel(train_data['path'].iloc[num])
        inputTensor = inputTensor.cuda()
        target = torch.tensor([[train_data['label'].iloc[num]]], dtype=torch.float32)
        target = target.cuda()
        outPut = cnnNetVoice(inputTensor)
        loss = lossFu(outPut, target)
        optimizer.zero_grad()
        loss.backward()#反向传播
        optimizer.step()
        trainStep += 1
        #  打印实时进度
        progress = (num + 1) / len(train_data) * 100
        sys.stdout.write(f"\r本轮训练进度: {num + 1}/{len(train_data)} ({progress:.2f}%)  当前 loss: {loss.item():.4f}")
        sys.stdout.flush()  # 刷新输出缓冲区
        if(trainStep % 100 == 0):#100次记录一次
            writer.add_scalar("100尺度下训练损失随次数变换", loss.item(), trainStep)#写入tensor
        if(trainStep % 1000 == 0):
            writer.add_scalar("1000尺度下训练损失随次数变换", loss.item(), trainStep)  # 写入tensor

    with (torch.no_grad()):#测试集验证
        for num in range(len(test_data)):
            inputTensor = load_npy_mel(test_data['path'].iloc[num])
            inputTensor = inputTensor.cuda()
            target = torch.tensor([[test_data['label'].iloc[num]]], dtype=torch.float32) #真实值
            target = target.cuda()
            outPut = cnnNetVoice(inputTensor)#预测值
            loss = lossFu(outPut, target)
            pred_label = (torch.sigmoid(outPut) > 0.5).float()
            #正确的个数
            correct_total += (pred_label == target).sum().item()

            testStep += 1
            progress = (num + 1) / len(test_data) * 100
            sys.stdout.write(f"\r本轮测试: {num + 1}/{len(test_data)} ({progress:.2f}%)  当前 loss: {loss.item():.4f}")
            sys.stdout.flush()  # 刷新输出缓冲区
            if(testStep % 100 == 0):
                writer.add_scalar("100尺度测试损失随次数变化", loss.item(), testStep)
            if(testStep % 1000 == 0):
                writer.add_scalar("1000尺度测试损失随次数变化", loss.item(), testStep)
    print("第"+str(i)+"轮的准确率 " + str(correct_total/len(test_data)*100) + "%" )
    torch.save(cnnNetVoice.state_dict(), "cnnNetVoice_{}.pth".format(i))
writer.close()

# tensorboard --logdir=./log
# print(train_data.head())
# print(load_npy_mel(train_data['path'].iloc[1]))
# print(len(train_data))
# target= train_data['label'].iloc[0]
# print(type(target))
#总训练集进行10次
# inputTensor = load_npy_mel(train_data['path'].iloc[0])
# target = train_data['label'].iloc[0]
# outPut = cnnNetVoice(inputTensor)
# print(type(outPut))