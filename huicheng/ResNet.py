# load CIFAR-100 dataset
from torchvision import datasets
import torchvision.models as models
from torchvision.transforms import transforms
# import neccessary libraries
import os
import sys
import json
import requests
from tqdm import tqdm
import time
import datetime
import logging
import logging.handlers
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from datasets import load_dataset
import torch.nn.parallel
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.layer1 = self.make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 256, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out) 
        out = self.fc(out)
        return out
        
    def evaluate(self, test_loader, criterion, use_cuda):
        # calculate the accuracy on the test set
        model.eval()
        test_loss = 0.0
        class_correct = list(0. for i in range(100))
        class_total = list(0. for i in range(100))
        for data, target in tqdm(test_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not use_cuda else np.squeeze(correct_tensor.cpu().numpy())
            for i in range(len(target.data)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        test_loss = test_loss/len(test_loader.dataset)
        for i in range(10):
            if class_total[i] > 0:
                # log accuracy of each class
                wandb.log({"acc_{}".format(classes[i]): class_correct[i] / class_total[i]})
                #print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))  
        #wandb log the average acc
        wandb.log({"acc": np.sum(class_correct) / np.sum(class_total)})
        # print average acc
        print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
        

    def train_model(model, train_loader, valid_loader, epochs, optimizer, criterion, use_cuda, save_path):
        valid_loss_min = np.Inf
        count = 0
        for epoch in tqdm(range(1, epochs+1)):
            train_loss = 0.0
            valid_loss = 0.0
            model.train()
            for data, target in train_loader:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)
            model.eval()
            for data, target in valid_loader:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
            model.evaluate( valid_loader, criterion, use_cuda)
            wandb.log({"training_loss": train_loss, "val_loss": valid_loss})  

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss
                count = 0
            else :
                count = count + 1
                if count > 4:
                    break




    def predict(model, test_loader, use_cuda):
        model.eval()
        test_preds = torch.LongTensor()
        use_cuda = torch.cuda.is_available()
        for i, data in tqdm(test_loader):
            if use_cuda:
                data = data.cuda()
            output = model(data)
            preds = output.cpu().data.max(1, keepdim=True)[1]
            test_preds = torch.cat((test_preds, preds), dim=0)
        return test_preds
