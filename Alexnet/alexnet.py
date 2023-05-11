import torch
import torchvision
from torch import nn

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import AlexNet_Weights



#define alexnet

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        #扁平层
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
            nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),nn.Flatten(),
            nn.Linear(6400,4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )

     # x为传入数据
    def forward(self, x):  # 前向传播
       # x先经过碾平变为1维
  #      x = self.flatten(x)
         # 随后x经过linear_relu_stack
        logits = self.linear_relu_stack(x)
         # 输出logits
        return logits

#预处理
preprocess = transforms.Compose([
 #   transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize([0.1307,], [0.3081,])
])


# 定义训练函数，需要
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X = preprocess(X)
     #   print(X.shape)
        X, y = X.cuda(), y.cuda()
     #   print(X.shape)
    #    X = preprocess(X)
        # 得到预测的结果pred
        pred = model(X)

        # 计算预测的误差
        # print(pred,y)
        loss = loss_fn(pred, y)

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练100次，输出一次当前信息
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    print("size = ",size)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X = preprocess(X)
       #     print(X.shape)
            X, y = X.cuda(), y.cuda()
      #      X = preprocess(X)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("correct = ",correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__=='__main__':
    # MNIST dataset
    batch_size = 100
    train_dataset = datasets.MNIST(root = './ml/pymnist',
                                   train = True,
                                   transform = torchvision.transforms.ToTensor(),
                                   download= True)
    test_dataset = datasets.MNIST(root = './ml/pymnist',
                                  train = False,
                                  transform = torchvision.transforms.ToTensor(),
                                  download = True)

    #加载数据
    train_loader = DataLoader(dataset= train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset= test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    #检验数据
    print('train_data    ', train_dataset.data.size())
    print('train_targets ', train_dataset.targets.size())
    print('test_data     ', test_dataset.data.size())
    print('test_targets  ', test_dataset.targets.size())

    for X, y in test_loader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break


    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # 调用刚定义的模型，将模型转到GPU（如果可用）
    model = AlexNet().to(device)

 #   print(model)

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()
    #优化器的选择
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 初始学习率

    epoch = 0
    epochs = 10
    for epoch in range(epochs):
        print(f"----epoch {epoch+1} ----")
        train(train_loader, model, loss_fn , optimizer)
        test(test_loader, model)
        print("Done!")
    torch.save(model.state_dict(),"model.pth")
    model.load_state_dict(torch.load("model.pth"))



