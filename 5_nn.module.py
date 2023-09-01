import torch
from torch import nn

#所有创建自己的网络结构都要继承nn.Module，forward函数在调用类的时候会自动执行
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)