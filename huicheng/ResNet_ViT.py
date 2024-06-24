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

import torch
import torch.nn as nn
import torch.nn.functional as F


import wandb
import random
from torch.utils.data import DataLoader

wandb.init(
    project="DL_Classification_CIFAR-100",
    config={
    "learning_rate": 5e-3,
    "architecture": "ResNet_ViT",
    "dataset": "CIFAR-100",
    "epochs": 50,
    }
)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=129, patch_size=8, emb_size=768):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # print("In Patch Embedding forward method")
        # print("before patch embedding size:", x.size())
        x = self.proj(x)  # 卷积操作
        # print("after proj size:", x.size())

        x = x.transpose(1, 2)  # 将特征维度与序列维度交换
        # print("after transpose size:", x.size())
        # print("after patch embedding")
        return x




class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=12, n_heads=12, ff_hidden=3072):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=ff_hidden, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        x = self.encoder(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32*32, patch_size=8, emb_size=768):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.position_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        self.transformer = TransformerEncoder(emb_size=emb_size)
        self.to_cls_token = nn.Identity()

    def forward(self, x):
        # print("In Vision Transformer forward method")
        #b = x.shape[0]
        #x = self.patch_embedding(x)
        #cls_token = self.cls_token.expand(b, -1, -1)  # Adjusting cls_token to match batch size of x
        #print("x size:", x.size())
        #print("cls_token size:", cls_token.size())      
        #x = torch.cat((cls_token, x), dim=1)  # [B, 1+N, E]
        #print("After cat size:", x.size())
        #x = x + self.position_embedding[:x.size(1), :]
        #print("After adding position embedding size:", x.size())
        x = self.transformer(x)
        #x = self.to_cls_token(x[:, 0])
        # print("After Vision Transformer")

        return x

class HybridResNetViT(nn.Module):
    def __init__(self, transformer, num_classes=100, emb_size=768, patch_size=8):
        super(HybridResNetViT, self).__init__()
        resnet = ResNet(ResidualBlock, [5, 6, 7, 8])
        resnet.load_state_dict(torch.load('ResNet_baseline_model.pt'))
        for param in resnet.parameters():
            param.requires_grad = True
        self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        self.transformer = transformer
        self.classifier = nn.Linear(emb_size, num_classes)
        
        # 新增：用于调整patches尺寸的全连接层
        self.patch_dim_transform = nn.Linear(16, emb_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        num_patches = ((32*32 // patch_size) ** 2)
        self.position_embedding = nn.Parameter(torch.randn(num_patches + 1, emb_size))
        self.to_cls_token = nn.Identity()

    def forward(self, x):
        # print("Input Shape:", x.shape)
        features = self.resnet_feature_extractor(x)
        # print("Feature Map Shape:", features.shape)
        patches = self._preprocess_features(features)
        # print("before transform Patches Shape:", patches.shape)
        patches = self.patch_dim_transform(patches)  # 调整patches的维度
        # print("after transform Patches Shape:", patches.shape)
        
        batch_size = patches.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # print("cls_token size:", cls_token.size())
        
        x = torch.cat((cls_token, patches), dim=1)
        # print("After cat size:", x.size())
        
        x = x + self.position_embedding[:x.size(1), :]
        # print("After adding position embedding size:", x.size())
        
        x = self.transformer(x)
        # print("After transformer size:", x.size())
        x = self.to_cls_token(x[:, 0])
        # print("After to_cls_token size:", x.size())
        x = self.classifier(x)
        # print("After classifier size:", x.size())
        
        return x
    
    def _preprocess_features(self, features):
        batch_size, channels, height, width = features.size()
        patch_size = min(height, width, 8)  # Ensure the patch size does not exceed feature dimensions

        if height < patch_size or width < patch_size:
            raise ValueError("Feature map size is too small for the given patch size.")

        features = features.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        features = features.contiguous().view(batch_size, channels, -1, patch_size * patch_size)
        features = features.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, patch_size * patch_size)

        return features


    def evaluate(self, test_loader, criterion, use_cuda):
        self.eval()
        test_loss = 0.0
        class_correct = list(0. for i in range(100))
        class_total = list(0. for i in range(100))
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = self(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, 1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not use_cuda else np.squeeze(correct_tensor.cpu().numpy())
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        test_loss /= len(test_loader.dataset)
        for i in range(100):
            if class_total[i] > 0:
                # log accuracy of each class
                wandb.log({"acc_{}".format(classes[i]): class_correct[i] / class_total[i]})
                #print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
               print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))  
        overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
        wandb.log({"acc": overall_accuracy})
        print(f'Test Loss: {test_loss:.6f}')
        print(f'Test Accuracy (Overall): {overall_accuracy:.2f}% ({np.sum(class_correct)}/{np.sum(class_total)})')

    def train_model(self, train_loader, valid_loader, epochs, optimizer, criterion, use_cuda, save_path):
        valid_loss_min = np.Inf
        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0
            self.train()
            cnt = 0
            for data, target in tqdm(train_loader):
                cnt = cnt + 1
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)  # Add gradient clipping here
                optimizer.step()
                # print(f'Batch: {cnt} \tTraining Loss: {loss.item():.6f}')
                train_loss += loss.item() * data.size(0)
            self.evaluate(valid_loader, criterion, use_cuda)  # Changed to self.evaluate
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            wandb.log({"training_loss": train_loss, "val_loss": valid_loss})  
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
            if valid_loss <= valid_loss_min:
                print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...')
                torch.save(self.state_dict(), save_path)
                valid_loss_min = valid_loss

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
