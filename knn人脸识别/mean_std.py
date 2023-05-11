import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from Dataloader import LoadData


def batch_mean_std(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b*h*w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])

        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

    return mean, std


data_path = "./dataset/"

transform_img = transforms.Compose([
    transforms.ToTensor(),
])

#image_data = datasets.CIFAR10(root='dataset/', train=True,
# 							   transform=transforms.ToTensor(),
#                              download=True)

train_dataset = LoadData("train.txt", True)
test_dataset = LoadData("test.txt", False)

train_loader = DataLoader(dataset=train_dataset, batch_size=240, num_workers=0, pin_memory=True,
                              shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=160, num_workers=0, pin_memory=True)


batch_size = 240
loader = DataLoader(train_dataset, batch_size=batch_size,
                    num_workers=0)


mean, std = batch_mean_std(loader)
print("mean and std: ", mean, std)


