import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#定义tansform 数据转换处理方式 转成张量和
'''
# 标准化

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transforms.Normalize(mean, std) 的计算公式：
input[channel] = (input[channel] - mean[channel]) / std[channel]
for example:
from torchvision import models, transforms

# 迁移学习，预训练模型
net = models.resnet18(pretrained=True)

# 标准化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 数据转换
image_transform = transforms.Compose([
    # 将输入图片resize成统一尺寸
    transforms.Resize([224, 224]),
    # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
    transforms.ToTensor(),
    # 标准化处理-->转换为标准正太分布，使模型更容易收敛
    normalize
])
'''
normalize=transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
transform=transforms.Compose([transforms.ToTensor(),normalize])

trainset=torchvision.datasets.CIFAR10(root='D:\data',train=True,download='False',transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='D:\data',train=False,download='False',transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=2)

classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        #输入通道数3,输出通道数6 卷积核大小kernel_size=5
        self.Conv1=nn.Conv2d(3,6,5)
        #kernal=2*2 stride=(2,2)
        self.subsample=nn.MaxPool2d(2,2)
        self.Conv2=nn.Conv2d(6,16,5)
        #dim=16*5*5->dim=120     
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    
    def forward(self,x):
    #32*32*3 ->28*28*6->14*14*6->10*10*16->5*5*16
        x=self.subsample(F.relu(self.Conv1(x)))
        x=self.subsample(F.relu(self.Conv2(x)))
    #5*5*16->400
    #16*5*5一定要写对
        x=x.view(-1,16*5*5)
    #16*5*5->120 120->84 84->10
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        y=self.fc3(x)
        return y
net=LeNet()
Critertion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
'''
numbers = [45, 22, 14, 65, 97, 72]
for i, num in enumerate(numbers):
    if num % 3 == 0 and num % 5 == 0:
        numbers[i] = 'fizzbuzz'
    elif num % 3 == 0:
        numbers[i] = 'fizz'
    elif num % 5 == 0:
        numbers[i] = 'buzz'
print(numbers)
'''
epoch_num=10
for epoch in range(epoch_num):
    running_loss=0.0
    for i,data in enumerate(trainloader):
        
        inputs,labels=data
        ##      input.size() <- torch.Size([4, 3, 32, 32]) batch_size=4 4张 每张3通道 32*32 
        #print('inputs_size:',inputs.size())
        #print(labels) # tensor([3, 3, 8, 6]) 是一个四维张量 
        #梯度初始化为0
        optimizer.zero_grad()
        #前向传播得到输出
        outputs=net(inputs)
        #交叉熵损失
        loss=Critertion(outputs,labels)
        #后向传播
        loss.backward()
        #优化函数更新参数 梯度下降SGD <- optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
        optimizer.step()
        #一个元素张量可以用x.item()得到元素值，我理解的就是一个是张量，一个是元素。
        running_loss+=loss.item()
        #print('loss','\n',loss)
        #print('loss.item()','\n',loss.item())
        if(i%2000 == 1999):
            print('[&d, %5d] loss:%0.3f'%
                 (epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print('finish training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images,labels=data
        outputs=net(images)
        _,predicted = torch.max(outputs.data,1)
        total+=labels.size(0)
        correct += (predicted == labels).sum().item()
print(100*correct/total)
