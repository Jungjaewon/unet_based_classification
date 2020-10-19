import torch.nn as nn
import torch

if __name__ == '__main__':
     down_sample = nn.AdaptiveAvgPool2d((1,1))
     tensor = torch.Tensor(1,128,16,16)

     print(down_sample(tensor).size())