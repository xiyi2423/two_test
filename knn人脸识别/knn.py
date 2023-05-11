import torch
import operator
import numpy as np
import time

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from Dataloader import LoadData, transform_BZ

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Knn:
    def __init__(self):  # 什么函数？
        pass

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, k, dis, X_test):
        assert (dis == 'E' or dis == 'M')
        num_test = X_test.shape[0]
        label_list = []

        # 使用欧氏距离作为距离度量
        if dis == 'E':
            for i in range(num_test):
                # 计算第 i 个测试样本与训练数据中所有点的欧氏距离
                distances = np.sqrt(
                    np.sum((self.X_train - X_test[i]) ** 2,
                           axis=1))
                # print(distances)
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                classCount = {}
                for j in topK:
                    classCount[self.Y_train[j]] = \
                        classCount.get(self.Y_train[j], 0) + 1
                # print(classCount)
                sortedClassCount = sorted(classCount.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True)
                label_pred = sortedClassCount[0][0]
                label_list.append(label_pred)
                # print(label_pred)
            return np.array(label_list)
        elif dis == 'M':
            for i in range(num_test):
                # 计算第 i 个测试样本与训练数据中所有点的欧氏距离
                distances = np.sum(np.abs(self.X_train - X_test[i]),
                                   axis=1)
                # print(distances)
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                classCount = {}
                for j in topK:
                    classCount[self.Y_train[j]] = \
                        classCount.get(self.Y_train[j], 0) + 1
                # print(classCount)
                sortedClassCount = sorted(classCount.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True)
                label_pred = sortedClassCount[0][0]
                label_list.append(label_pred)
                # print(label_pred)
            return np.array(label_list)




if __name__=='__main__':


    #给训练集和测试集分别创造一个数据集加载器
    train_dataset = LoadData("train.txt",True)
    test_dataset = LoadData("test.txt",False)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=240,num_workers=4, pin_memory=True,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=160, num_workers=4, pin_memory=True)

    for data in train_dataloader:
        X_train,Y_train = data
   #     X_train = transform_BZ(X_train)
        X_train = X_train.flatten(1)


    for data in test_dataloader:
        X_test,Y_test = data
  #      X_test = transform_BZ(X_test)
        X_test = X_test.flatten(1)


    X_train,Y_train = X_train.numpy(),Y_train.numpy()


    X_test,Y_test = X_test.numpy(),Y_test.numpy()


    print(X_test.shape)
    print(X_train.shape)

    num_test = Y_test.shape[0]
    KnnClassifier = Knn()
    KnnClassifier.fit(X_train, Y_train)

    start_time = time.time()
    print('%.3f' % start_time)
    print(time.asctime(time.localtime(start_time)))

    Y_test_pred = KnnClassifier.predict(9, 'M', X_test)

    end_time = time.time()
    print('%.3f' % end_time)
    print(time.asctime(time.localtime(end_time)))

    duration = end_time - start_time
    print('共耗时 %.3f 秒' % (duration))

    num_correct = np.sum(Y_test_pred == Y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct, accuracy is %f' \
                % (num_correct, num_test, accuracy))



