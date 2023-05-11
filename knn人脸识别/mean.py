import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from Dataloader import LoadData

batch_size=240 #这里是为了后面一次取出所有的数据
# transform=transforms.Compose([transforms.ToTensor(),
#                               transforms.Normalize((0.1307,),(0.3081,))])
transform=transforms.Compose([transforms.ToTensor()]) #不对数据进行标准化

#加载数据
train_dataset = LoadData("train.txt", True)
test_dataset = LoadData("test.txt", False)

train_loader = DataLoader(dataset=train_dataset, batch_size=240, num_workers=4, pin_memory=True,
                              shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=160, num_workers=4, pin_memory=True)

#取出加载在DataLoader中的数据，因为batch_size就是训练集的样本数目，所以一次就取完了所有训练数据
for batch_idx, data in enumerate(train_loader, 0):
    inputs, targets = data #inpus为所有训练样本
    x=inputs.view(-1,28*28) #将（6000，1，28，28）大小的inputs转换为（60000，28*28）的张量
    x_std=x.std().item() #计算所有训练样本的标准差
    x_mean=x.mean().item() #计算所有训练样本的均值

print('均值mean为:'+str(x_mean))
print('标准差std为:'+str(x_std))


