
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize

BATCH_SIZE = 256

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
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1 = nn.Linear(1*28*28,28)
        self.fc2 = nn.Linear(28,10)

    def forward(self,input):

        #1修改形状
        x = input.view([input.size(0),1*28*28])
        #2全连接
        x = self.fc1(x)
        #3激活函数,形状无变化
        x = F.relu(x)
        #4输出层
        out = self.fc2(x)
        return F.log_softmax(out)

model = MnistModel()
optimizer = Adam(model.parameters(),lr = 0.001)

def train(epoch):
    data_loader = get_dataloader()
    for idx,(input,traget) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)     #调用模型
        loss = F.nll_loss(output,traget)#得到损失函数
        loss.backward()    #反向传播
        optimizer.step()
        if idx%100 == 0:
            print(loss.item())


if __name__ == '__main__':
    for i in range(3):
        train(i)