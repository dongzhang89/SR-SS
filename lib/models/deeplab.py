from torch import nn
import torch
import sys
sys.path.insert(0, "../")
from models.resnet import *
from utils.config import cfg
from collections import OrderedDict

class ClassLayer(nn.Module):
    """docstring for ClassLayer"""
    def __init__(self, in_channel, num_classes=21):
        super(ClassLayer, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, stride=1))
        self.classification = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=self.num_classes, out_features=self.num_classes)

    def forward(self, x):
        x = self.layer(x)
        pred = self.classification(x)
        pred = pred.view(pred.shape[0], -1)
        pred = torch.sigmoid(self.fc(pred))
        return x, pred

class DeepLab(nn.Module):
    """"Deeplab for semantic segmentation """
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        self.pretrained_model = cfg.TRAIN.PRETRAINED_MODEL

    def _init_module(self):
        # Define the network
        self.Scale = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=self.num_classes)
        self.classlayer1 = ClassLayer(in_channel=64, num_classes=self.num_classes)
        self.classlayer2 = ClassLayer(in_channel=256, num_classes=self.num_classes)
        self.classlayer3 = ClassLayer(in_channel=512, num_classes=self.num_classes)
        self.classlayer4 = ClassLayer(in_channel=1024, num_classes=self.num_classes)
        self.classlayer5_1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.classlayer5_2 = nn.Linear(in_features=self.num_classes, out_features=self.num_classes)

        # Fix BatchNorm
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        self.Scale.apply(set_bn_fix)
        # Fix blocks

    def forward(self, input):
        input_size = input.size()
        h, w = input.size()[2:]
        interp = nn.UpsamplingBilinear2d(size=(h, w))

        out_list = []
        pred_list = []
        x, x_list = self.Scale(input)
        # print("scale1: ", x.size())
        pred = self.classlayer5_1(x)
        # print("classlayer5_1: ", pred)
        pred = self.classlayer5_2(pred.view(pred.shape[0], -1))
        pred = torch.sigmoid(pred)
        # print("pred: ", pred.size())
        # print("****************************")
        x = interp(x)
        out_list.append(x)
        pred_list.append(pred)

        x1, pred1 = self.classlayer1(x_list[0])
        x1 = interp(x1)
        out_list.append(x1)
        pred_list.append(pred1)
        x2, pred2 = self.classlayer2(x_list[1])
        x2 = interp(x2)
        out_list.append(x2)
        pred_list.append(pred2)
        x3, pred3 = self.classlayer3(x_list[2])
        x3 = interp(x3)
        out_list.append(x3)
        pred_list.append(pred3)
        x4, pred4 = self.classlayer4(x_list[3])
        x4 = interp(x4)
        out_list.append(x4)
        pred_list.append(pred4)

        return out_list, pred_list

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        for m in self.Scale.layer5.conv2d_list:
            normal_init(m, 0, 0.01)

    def create_architecture(self):
        self._init_module()
        self._init_weights()

if __name__ == '__main__':

    net = DeepLab(num_classes=21)
    print(net)
    net.create_architecture()
    print(net)