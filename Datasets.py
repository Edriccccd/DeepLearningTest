import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize

BATCH_SIZE = 128

#1.Datas
def get_dataloader(train = True):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,),std=(0.3081,))
    ])
    dataset = MNIST(root="./data",train=train,transform = transform_fn)
    data_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    return data_loader

#2.Model
class MnistModel(nn.Moudle):
    def __init__(self)
        super(MnistModel,self).__init__()
        self.fcl = nn.liner(1*28*28,28)
        self.fc2 = nn.Linear(28,10)
    def forward(self,imput):
        #1修改形状
        x = input.view([input.size(0),1*28*28])
        #2全连接
        x = self.fcl(x)
        #3激活函数,形状无变化
        x = F.relu(x)
        #4输出层
        out = self.fc2(x)
        return F.log_softmax(out)

def train(epoch)
    model = MnistModel()