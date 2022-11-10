import numpy as np
import os.path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
#1.Datas
def get_dataloader(train = True,batch_size = BATCH_SIZE):
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
        return F.log_softmax(out,dim = -1)

model = MnistModel()
optimizer = Adam(model.parameters(),lr = 0.001)

if os.path.exists("./model/model.pk1"):
    model.load_state_dict(torch.load("./model/model.pk1"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pk1"))

def train(epoch):
    data_loader = get_dataloader()
    for idx,(input,traget) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)     #调用模型
        loss = F.nll_loss(output,traget)#得到损失函数
        loss.backward()    #反向传播
        optimizer.step()
        if idx%10 == 0:
            print(epoch,idx,loss.item())

        if idx%100 == 0:
            torch.save(model.state_dict(),"./model/model.pk1")
            torch.save(optimizer.state_dict(), "./model/optimizer.pk1")

def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train = False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,traget) in enumerate(test_dataloader):
        with torch.no_grad():
            output =  model(input)
            cur_loss = F.nll_loss(output,traget)
            loss_list.append(cur_loss)
            pred = output.max(dim = -1)[-1]
            cur_acc = pred.eq(traget).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))



if __name__ == '__main__':
    # for i in range(3):
    #     train(i)
    # loader =  get_dataloader(train = False)
    # for input,lable in loader:
    #     print(lable.size())
    #     break
    test()