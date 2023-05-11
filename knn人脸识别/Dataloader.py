import numpy as np
import torch
from PIL import Image

import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset

# 数据归一化与标准化

# 图像标准化
transform_BZ= transforms.Normalize(
    mean=[1.3102,1.3102,1.3102],# 取决于数据集
    std=[0.5000,0.5000,0.5000]
)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        #训练与测试数据的归一化与标准化
        self.train_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transform_BZ
            ])
        self.val_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transform_BZ
            ])
    #加载得到图像信息
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info
    #填充黑色操作
    def padding_black(self, img):
        w, h  = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    #True为将“train.txt"文件数据看作训练集对待
    train_dataset = LoadData("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(image)
        print(label)

   # print('train_data    ', train_dataset.img.size())
   # print('train_targets ', train_dataset.label.size())

    # test_dataset = Data_Loader("test.txt", False)
    # print("数据个数：", len(test_dataset))
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=10,
    #                                            shuffle=True)
    # for image, label in test_loader:
    #     print(image.shape)
    #     print(label)