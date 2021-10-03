import torch
import torch.nn as nn


class LeNet(nn.Module):  # nn.Module, 定义神经网络必须继承的模块， 框架规定的形式
    def __init__(self, channel=3, hidden=768, num_classes=10):  # 假设输入cifar10数据集， 默认3通道， 隐层维度为768， 分类为10
        super(LeNet, self).__init__()  # 继承pytorch神经网络工具箱中的模块
        act = nn.ReLU  # 激活函数为Sigmoid
        # nn.Sequential: 顺序容器。 模块将按照在构造函数中传递的顺序添加到模块中。 或者，也可以传递模块的有序字典
        self.body = nn.Sequential(  # 设计神经网络结构，对于nn.Sequential.Preference : https://zhuanlan.zhihu.com/p/75206669
            # 设计输入通道为channel，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为1的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act()
        )
        # 设计一个全连接映射层， 将hidden隐藏层映射到十个分类标签
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    # 设计前向传播算法
    def forward(self, x):
        out = self.body(x)  # 先经过nn.Sequential的顺序层得到一个输出
        out = out.view(out.size(0), -1)  # 将输出转换对应的维度
        out = self.fc(out)  # 最后将输出映射到一个十分类的一个列向量
        return out
