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


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()  # b 16,96,96深度、长、宽
        layer1 = nn.Sequential()
        layer1.add_module("conv1", nn.Conv2d(1, 16, 5, stride=3, padding=1))  # b 16 32 32深度、长、宽
        layer1.add_module("norm1", nn.BatchNorm2d(16))
        layer1.add_module("relu1", nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module("conv2", nn.Conv2d(16, 32, 5, stride=1))  # b 32 28 28深度、长、宽
        layer2.add_module("norm2", nn.BatchNorm2d(32))
        layer2.add_module("relu2", nn.ReLU(True))
        layer2.add_module('f_drop1', nn.Dropout2d())
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

    def forward(self, input_x):
        conv1 = self.layer1(input_x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)  # 卷积网络(高维数据)
        conv4 = self.layer4(conv3)
        return conv4


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        layer5 = nn.Sequential()
        layer5.add_module("fc1", nn.Linear(128 * 5 * 5, 1024))  # 128*5*5=3200
        layer5.add_module("fc_nb", nn.BatchNorm1d(1024))
        layer5.add_module("fc_relu1", nn.ReLU(True))
        layer5.add_module("fc2", nn.Linear(1024, 128))
        layer5.add_module('c_drop1', nn.Dropout2d())
        layer5.add_module("fc_nb1", nn.BatchNorm1d(128))
        layer5.add_module("fc_relu2", nn.ReLU(True))
        layer5.add_module("fc3", nn.Linear(128, 2))
        layer5.add_module('c_softmax', nn.LogSoftmax(dim=1))
        self.layer5 = layer5

    def forward(self, x):
        fc_input = x.view(x.size(0), -1)  # 高维数据 ‘压’成 低维数据
        fc_out = self.layer5(fc_input)  # 全连接层(低维数据)
        return fc_out


class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)  # 保持x维度不变

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha  # grad_output.neg()相当于梯度反转
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=128 * 5 * 5, hidden_dim=2048):
        super(Discriminator, self).__init__()
        d_layer = nn.Sequential()
        d_layer.add_module("fd1", nn.Linear(input_dim, hidden_dim))  # 128*4*4=2048
        d_layer.add_module("fd_nb", nn.BatchNorm1d(hidden_dim))
        d_layer.add_module("fd_relu1", nn.ReLU(True))
        d_layer.add_module("fd2", nn.Linear(hidden_dim, 100))  # 128*4*4=2048
        d_layer.add_module("fd_nb2", nn.BatchNorm1d(100))
        d_layer.add_module("fd_relu2", nn.ReLU(True))
        d_layer.add_module("fd3", nn.Linear(100, 2))
        d_layer.add_module("sig", nn.LogSoftmax(dim=1))
        self.d_layer = d_layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 高维数据 ‘压’成 低维数据
        x = self.d_layer(x)
        return x


class DANN(nn.Module):
    def __init__(self, device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor().to(self.device)
        self.classifier = Classifier().to(self.device)
        self.domain_classifier = Discriminator().to(self.device)
        self.GRL = GRL()

    def forward(self, input_data, alpha=1, stop=False):  # stop是让反向传播提前停止，节省内存
        input_data = input_data.expand(len(input_data), 1, 96, 96)  # rgb图用len(input_data), 3, 28, 28
        feature = self.feature(input_data)
        # feature = feature.view(feature.size(0), -1)  可以写也可以不写，因为模型里面写了
        if stop:
            class_output = self.classifier(feature)
            x = GRL.apply(feature, alpha)
            domain_output = self.domain_classifier(x.detach())
        else:
            class_output = self.classifier(feature)
            x = GRL.apply(feature, alpha)
            domain_output = self.domain_classifier(x)
        return class_output, domain_output

def show_net(model_name):
    x = torch.rand(20, 1, 96, 96).to(DEVICE)
    torch.onnx.export(model_name, x, "DANN_EEG.onnx")
    netron.start("DANN_EEG.onnx")  # 输出网络结构

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = 'cpu'
batch_size = 20
learning_rate = 3e-4
# torch.random.manual_seed(517)#517 69
source_path = r".\9"
target_path = r".\4"
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize(mean=(0.5), std=(0.5))])
source_dataset = GetLoader_96(data_root=os.path.join(source_path, 'train'),
                          data_list=os.path.join(source_path, 'train9.txt'),
                          transform=data_transforms)
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
target_dataset = GetLoader_96(data_root=os.path.join(target_path, 'train'),
                         data_list=os.path.join(target_path, 'train4.txt'),
                         transform=data_transforms)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = GetLoader_96(data_root=os.path.join(target_path, 'test'),
                         data_list=os.path.join(target_path, 'test4.txt'),
                         transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = DANN(DEVICE).to(DEVICE)
c_criterion = nn.NLLLoss().to(DEVICE)  # 分类器损失函数
d_criterion = nn.NLLLoss().to(DEVICE)  # 鉴别器损失函数

opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

len_dataloader = min(len(source_dataset), len(target_dataset))
d_src = torch.zeros(batch_size).to(DEVICE)  # 源域数据标签
d_tar = torch.ones(batch_size).to(DEVICE)  # 目标域数据标签
num_epochs = 50
# show_net(model)
for epoch in range(num_epochs):
    i = 1
    model.train()
    p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    for (data_src, data_tar) in zip(enumerate(source_loader), enumerate(target_loader)):
        _, (x_src, y_src) = data_src
        _, (x_tar, _) = data_tar
        x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
        class_output, s_domain_output = model(input_data=x_src, alpha=alpha)  # 源域输入
        err_s_label = c_criterion(class_output, y_src)
        err_s_domain = d_criterion(s_domain_output, d_src.long())
        _, t_domain_output = model(input_data=x_tar, alpha=alpha)  # 目标域输入
        err_t_domain = d_criterion(t_domain_output, d_tar.long())  # 注意这个是源域标签，因为要欺骗
        err_domain = err_t_domain + err_s_domain
        err = (err_t_domain + err_s_domain) + err_s_label
        opt.zero_grad()
        err.backward()
        opt.step()

        i += 1
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, '
                  'domain_loss: {:.4f},total_loss: {:.4f} '
                  .format(epoch + 1, num_epochs, i + 1, len_dataloader // batch_size, err_s_label.item(),
                          err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item()))

    alpha = 0
    model.eval()
    n_correct = 0
    total = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(test_loader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _ = model(input_data=t_img, alpha=alpha)
            _, predicted = torch.max(class_output.data, 1)
            total += t_label.size(0)  # 计算所有的label数量
            n_correct += (predicted == t_label.squeeze(-1)).sum()  # 计算预测对的label数量
    print('Accuracy of the network on the %d test images: %d %%' % (total, (100 * torch.true_divide(n_correct, total))))
    if 100 * torch.true_divide(n_correct, total)>83 and epoch>40:
        torch.save(model.state_dict(), './model_state942.pth')
        break