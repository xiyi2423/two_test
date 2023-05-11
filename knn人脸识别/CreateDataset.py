'''
生成训练集和测试集，保存在txt文件中
'''

import os
import random

train_ratio = 0.6
test_ratio = 1 - train_ratio

rootdata = r"ORL/att_faces"

train_list, test_list = [],[]
data_list = []

class_flag=-1
for root,dirs,files in os.walk(rootdata):
    print(root)
    for i in range(len(files)):
        data_list.append(os.path.join(root,files[i]))

    for i in range(0,int(len(files)*train_ratio)):
        train_data = os.path.join(root,files[i])+'\t'+str(class_flag)+'\n'
        train_list.append(train_data)

    for i in range(int(len(files)*train_ratio),len(files)):
        test_data = os.path.join(root,files[i]) + '\t' + str(class_flag)+'\n'
        test_list.append(test_data)

    class_flag += 1

print(train_list)
random.shuffle(train_list)
random.shuffle(test_list)

with open('train.txt','w',encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('test.txt','w',encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(str(test_img))