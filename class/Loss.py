import torch
from torch import nn

class LossDegree(nn.Module):
    def __init__(self):
        super(LossDegree, self).__init__()

    def forward(self, predictions, labels):
        # 计算预测值和真实值之间的差
        differences = predictions - labels
        differences_adjusted = torch.fmod(differences + 180, 360) - 180
        # 计算差的平方
        squared_differences = differences_adjusted ** 2
        # 计算平方差的均值
        mean_squared_differences = squared_differences.mean()
        # 取平方根得到RMSE
        rmse = torch.sqrt(mean_squared_differences)
        return rmse