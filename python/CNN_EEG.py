# -*-coding:UTF-8-*-
from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import scipy.io as scio
import torch.utils.data as data
from PIL import Image
import os
import netron
import torch.onnx

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None, ):
        self.root = data_root
        self.transform = transform
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()
        self.n_data = len(data_list) // 6

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        width=96
        length=96
        # pic = Image.new('RGB',(96,96))
        pic=torch.zeros((3,width,length))
        label = 0
        for num in range(6 * item, 6 * item + 3):
            img_paths, labels = self.img_paths[num], self.img_labels[num]
            imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')
            # imgs = imgs.crop((114, 48, 792, 584))#单个图的
            # imgs = imgs.crop((114, 358, 792, 584))  # 329:585,115:792，三个图的
            # pic.paste(imgs, box=(0, num % 3 * 226))
            label += int(labels)
        # if self.transform is not None:
        #     pic = pic.resize((width, length))
        #     pic = self.transform(pic)
            if self.transform is not None:
                imgs = self.transform(imgs)
                pic[num % 3, ...] =imgs
        if label == 3:
            return pic, 0
        elif  label == 6:
            return pic, 1


    def __len__(self):
        return self.n_data


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # b 6,100,100深度、长、宽
        layer1 = nn.Sequential()
        layer1.add_module("conv1", nn.Conv2d(1, 16, 7, stride=3))  # b 16 32 32深度、长、宽
        layer1.add_module("norm1", nn.BatchNorm2d(16))
        layer1.add_module("relu1", nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module("conv2", nn.Conv2d(16, 32, 5, stride=1))  # b 32 28 28深度、长、宽
        layer2.add_module("norm2", nn.BatchNorm2d(32))
        layer2.add_module("relu2", nn.ReLU(True))
        layer2.add_module("pool2", nn.MaxPool2d(2, 2))  # b 32 14 14 深度、长、宽
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module("conv3", nn.Conv2d(32, 64, 3))  # b 64 12 12深度、长、宽
        layer3.add_module("norm3", nn.BatchNorm2d(64))
        layer3.add_module("relu3", nn.ReLU(True))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module("conv4", nn.Conv2d(64, 128, 3, stride=1))  # b 128 10,10深度、长、宽
        layer4.add_module("norm4", nn.BatchNorm2d(128))
        layer4.add_module("relu4", nn.ReLU(True))
        layer4.add_module("pool4", nn.MaxPool2d(2, 2))  # b 128 5 5 深度、长、宽
        self.layer4 = layer4

        # layer_add = nn.Sequential()
        # layer_add.add_module("conv_add", nn.Conv2d(128, 256, 3, stride=2))  # b 256 5 7深度、长、宽
        # layer_add.add_module("norm_add", nn.BatchNorm2d(256))
        # layer_add.add_module("relu_add", nn.ReLU(True))
        # self.layer_add = layer_add

        layer5 = nn.Sequential()
        layer5.add_module("fc1", nn.Linear(3200, 1024))  # 128*5*5=3200
        layer5.add_module("fc_relu1", nn.ReLU(True))
        layer5.add_module("fc2", nn.Linear(1024, 128))
        layer5.add_module("fc_relu2", nn.ReLU(True))
        layer5.add_module("fc3", nn.Linear(128, 2))
        layer5.add_module("sig", nn.LogSoftmax(dim=1))
        self.layer5 = layer5

    def forward(self, input_x):
        conv1 = self.layer1(input_x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)  # 卷积网络(高维数据)
        conv4 = self.layer4(conv3)
        # conv5 = self.layer_add(conv4)
        fc_input = conv4.view(conv4.size(0), -1)  # 高维数据 ‘压’成 低维数据
        fc_out = self.layer5(fc_input)  # 全连接层(低维数据)
        return fc_out


class GetLoader_96(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')
        label=0
        if self.transform is not None:
            imgs = self.transform(imgs)
            label += int(labels)

        if label == 1:
            return imgs, 0
        elif label == 2:
            return imgs, 1

    def __len__(self):
        return self.n_data


class SimpleCNN_96(nn.Module):
    def __init__(self):
        super(SimpleCNN_96, self).__init__()  # b 6,96,96深度、长、宽
        layer1 = nn.Sequential()
        layer1.add_module("conv1", nn.Conv2d(1, 16, 5, stride=3,padding=1))  # b 16 32 32深度、长、宽
        layer1.add_module("norm1", nn.BatchNorm2d(16))
        layer1.add_module("relu1", nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module("conv2", nn.Conv2d(16, 32, 5, stride=1))  # b 32 28 28深度、长、宽
        layer2.add_module("norm2", nn.BatchNorm2d(32))
        layer2.add_module("relu2", nn.ReLU(True))
        layer2.add_module("pool2", nn.MaxPool2d(2, 2))  # b 32 14 14 深度、长、宽
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module("conv3", nn.Conv2d(32, 64, 3))  # b 64 12 12深度、长、宽
        layer3.add_module("norm3", nn.BatchNorm2d(64))
        layer3.add_module("relu3", nn.ReLU(True))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module("conv4", nn.Conv2d(64, 128, 3, stride=1))  # b 128 10,10深度、长、宽
        layer4.add_module("norm4", nn.BatchNorm2d(128))
        layer4.add_module("relu4", nn.ReLU(True))
        layer4.add_module("pool4", nn.MaxPool2d(2, 2))  # b 128 5 5 深度、长、宽
        self.layer4 = layer4

        # layer_add = nn.Sequential()
        # layer_add.add_module("conv_add", nn.Conv2d(128, 256, 3, stride=2))  # b 256 5 7深度、长、宽
        # layer_add.add_module("norm_add", nn.BatchNorm2d(256))
        # layer_add.add_module("relu_add", nn.ReLU(True))
        # self.layer_add = layer_add

        layer5 = nn.Sequential()
        layer5.add_module("fc1", nn.Linear(3200, 3200))  # 128*5*5=3200
        layer5.add_module("fc_relu1", nn.ReLU(True))
        layer5.add_module("fc2", nn.Linear(3200, 128))
        layer5.add_module("fc_relu2", nn.ReLU(True))
        layer5.add_module("fc3", nn.Linear(128, 2))
        layer5.add_module("sig", nn.LogSoftmax(dim=1))
        self.layer5 = layer5

    def forward(self, input_x):
        conv1 = self.layer1(input_x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)  # 卷积网络(高维数据)
        conv4 = self.layer4(conv3)
        # conv5 = self.layer_add(conv4)
        fc_input = conv4.view(conv4.size(0), -1)  # 高维数据 ‘压’成 低维数据
        fc_out = self.layer5(fc_input)  # 全连接层(低维数据)
        return fc_out

def show_net(model_name):
    x = torch.rand(20, 1, 96, 96).to(DEVICE)
    torch.onnx.export(model_name, x, "CNN_EEG.onnx")
    netron.start("CNN_EEG.onnx")  # 输出网络结构

def kappa_cal(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = 'cpu'
batch_size = 20
learning_rate = 1e-3
# torch.random.manual_seed(517)#10（68) 43(68) 22O0(68) 517(70)
path = r".\5"
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize(mean=(0.5), std=(0.5))])
train_dataset = GetLoader_96(data_root=os.path.join(path, 'bump'),
                          data_list=os.path.join(path, 'train.txt'),
                          transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = GetLoader_96(data_root=os.path.join(path, 'bump'),
                         data_list=os.path.join(path, 'test.txt'),
                         transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
model = SimpleCNN_96().to(DEVICE)
criterion = nn.NLLLoss().to(DEVICE)
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 100
# show_net(model)
for epoch in range(num_epochs):
    model.train()
    kappa_matrix = np.zeros([2,2])
    for i, (images, labels) in enumerate(train_loader):  # enumrate
        images = images.to(DEVICE)  # 图片大小为28*28,为什么不用Variable(images.view(-1, 28*28))？因为这是卷积神经网络，不用把图片以行或列向量输入模型
        labels = labels.to(DEVICE)
        opt.zero_grad()  # zero the gradient buffer
        outputs = model(images)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (
                      epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                      loss.item()))  # .item()是返回数值Float32

    model.eval()
    n_correct = 0
    total = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(test_loader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output= model(t_img)
            _, predicted = torch.max(class_output.data, 1)
            for kappa_matrix_num in range(batch_size):#生成kappa矩阵
                kappa_matrix[predicted[kappa_matrix_num],t_label[kappa_matrix_num]]+=1
            total += t_label.size(0)  # 计算所有的label数量
            n_correct += (predicted == t_label.squeeze(-1)).sum()  # 计算预测对的label数量
    print('Accuracy of the network on the %d test images: %d %%' % (total, (100 * torch.true_divide(n_correct, total))))
    print('KAPPA of the network on the %d test images: %d %%' % (total, 100 * kappa_cal(kappa_matrix)))
# model.eval()
# n_correct = 0
# total = 0
# with torch.no_grad():
#     for _, (t_img, t_label) in enumerate(test_loader):
#         t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
#         class_output= model(t_img)
#         _, predicted = torch.max(class_output.data, 1)
#         total += t_label.size(0)  # 计算所有的label数量
#         n_correct += (predicted == t_label.squeeze(-1)).sum()  # 计算预测对的label数量
# print('Accuracy of the network on the %d test images: %d %%' % (total, (100 * torch.true_divide(n_correct, total))))