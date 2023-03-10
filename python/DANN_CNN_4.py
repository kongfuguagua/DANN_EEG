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
        elif label == 3:
            return imgs, 2
        elif label == 4:
            return imgs, 3

    def __len__(self):
        return self.n_data


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()  # b 16,96,96
        layer1 = nn.Sequential()
        layer1.add_module("conv1", nn.Conv2d(1, 16, 5, stride=3, padding=1))  # b 16 32 32
        layer1.add_module("norm1", nn.BatchNorm2d(16))
        layer1.add_module("relu1", nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module("conv2", nn.Conv2d(16, 32, 5, stride=1))  # b 32 28 28
        layer2.add_module("norm2", nn.BatchNorm2d(32))
        layer2.add_module("relu2", nn.ReLU(True))
        layer2.add_module('f_drop1', nn.Dropout2d())
        layer2.add_module("pool2", nn.MaxPool2d(2, 2))  # b 32 14 14
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module("conv3", nn.Conv2d(32, 64, 3))  # b 64 12 12
        layer3.add_module("norm3", nn.BatchNorm2d(64))
        layer3.add_module("relu3", nn.ReLU(True))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module("conv4", nn.Conv2d(64, 128, 3, stride=1))  # b 128 10,10
        layer4.add_module("norm4", nn.BatchNorm2d(128))
        layer4.add_module("relu4", nn.ReLU(True))
        layer4.add_module("pool4", nn.MaxPool2d(2, 2))  # b 128 5 5
        self.layer4 = layer4

    def forward(self, input_x):
        conv1 = self.layer1(input_x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        return conv4


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        layer5 = nn.Sequential()
        layer5.add_module("fc1", nn.Linear(128 * 5 * 5, 3200))  # 128*5*5=3200
        layer5.add_module("fc_nb", nn.BatchNorm1d(3200))
        layer5.add_module("fc_relu1", nn.ReLU(True))
        layer5.add_module("fc2", nn.Linear(3200, 128))
        layer5.add_module("fc_nb1", nn.BatchNorm1d(128))
        layer5.add_module("fc_relu2", nn.ReLU(True))
        layer5.add_module("fc3", nn.Linear(128, 4))
        layer5.add_module('c_softmax', nn.LogSoftmax(dim=1))
        self.layer5 = layer5

    def forward(self, x):
        fc_input = x.view(x.size(0), -1)  # ???????????? ???????????? ????????????
        fc_out = self.layer5(fc_input)  # ????????????(????????????)
        return fc_out


class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)  # ??????x????????????

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha  # grad_output.neg()?????????????????????
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
        x = x.view(x.size(0), -1)  # ???????????? ???????????? ????????????
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

    def forward(self, input_data, alpha=1, stop=False):  # stop?????????????????????????????????????????????
        input_data = input_data.expand(len(input_data), 1, 96, 96)  # rgb??????len(input_data), 3, 28, 28
        feature = self.feature(input_data)
        # feature = feature.view(feature.size(0), -1)  ???????????????????????????????????????????????????
        if stop:
            class_output = self.classifier(feature)
            x = GRL.apply(feature, alpha)
            domain_output = self.domain_classifier(x.detach())
        else:
            class_output = self.classifier(feature)
            x = GRL.apply(feature, alpha)
            domain_output = self.domain_classifier(x)
        return class_output, domain_output


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
batch_size = 24
learning_rate = 1e-3
# torch.random.manual_seed(517)#517 69
source_path = r".\3"
target_path = r".\1"
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize(mean=(0.5), std=(0.5))])
source_dataset = GetLoader_96(data_root=os.path.join(source_path, 'bump'),
                          data_list=os.path.join(source_path, 'train.txt'),
                          transform=data_transforms)
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
target_dataset = GetLoader_96(data_root=os.path.join(target_path, 'bump'),
                         data_list=os.path.join(target_path, 'train.txt'),
                         transform=data_transforms)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = GetLoader_96(data_root=os.path.join(target_path, 'bump'),
                         data_list=os.path.join(target_path, 'test.txt'),
                         transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = DANN(DEVICE).to(DEVICE)
c_criterion = nn.NLLLoss().to(DEVICE)  # ?????????????????????
d_criterion = nn.NLLLoss().to(DEVICE)  # ?????????????????????

opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

len_dataloader = min(len(source_dataset), len(target_dataset))
d_src = torch.zeros(batch_size).to(DEVICE)  # ??????????????????
d_tar = torch.ones(batch_size).to(DEVICE)  # ?????????????????????
num_epochs = 100
# show_net(model)
for epoch in range(num_epochs):
    i = 1
    kappa_matrix = np.zeros([4, 4])
    model.train()
    p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    for (data_src, data_tar) in zip(enumerate(source_loader), enumerate(target_loader)):
        _, (x_src, y_src) = data_src
        _, (x_tar, _) = data_tar
        x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
        class_output, s_domain_output = model(input_data=x_src, alpha=alpha)  # ????????????
        err_s_label = c_criterion(class_output, y_src)
        err_s_domain = d_criterion(s_domain_output, d_src.long())
        _, t_domain_output = model(input_data=x_tar, alpha=alpha)  # ???????????????
        err_t_domain = d_criterion(t_domain_output, d_tar.long())  # ?????????????????????????????????????????????
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
            total += t_label.size(0)  # ???????????????label??????
            n_correct += (predicted == t_label.squeeze(-1)).sum()  # ??????????????????label??????
            for kappa_matrix_num in range(batch_size):#??????kappa??????
                kappa_matrix[predicted[kappa_matrix_num],t_label[kappa_matrix_num]]+=1
    print('Accuracy of the network on the %d test images: %d %%' % (total, (100 * torch.true_divide(n_correct, total))))
    print('KAPPA of the network on the %d test images: %d %%' % (total, 100 * kappa_cal(kappa_matrix)))