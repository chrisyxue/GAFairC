import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import types
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from FairBatchSampler import CustomDataset, FairBatch
from pytorch_transformers import AdamW, WarmupLinearSchedule


class Fair_Batch_Module(nn.Module):
    def __init__(self,in_dim):
        super(Fair_Batch_Module,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim*2),
            nn.ReLU(),
            # nn.Linear(in_dim*2,in_dim),
            # nn.ReLU(),
            nn.Linear(in_dim*2,1)
        )

    def forward(self, x):
        x=self.net(x)
        return x

def run_epoch(epoch, model, optimizer, criterion, train_loader, train_data):
    loss_mean = 0
    for batch_idx, (data, target, z) in enumerate(train_loader):
        # print(batch_idx)
        optimizer.zero_grad()
        y_pre = model(data)
        loss = criterion((F.tanh(y_pre.squeeze())+1)/2,(target.squeeze()+1)/2)
        loss.backward()
        optimizer.step()
        loss_mean += loss.item()
    loss_mean = loss_mean/(batch_idx)
    # print('loss: ',loss)

def train(X,y,saIndex,saValue,target_fairnes='eqodds'):
    model = Fair_Batch_Module(in_dim=X.shape[-1])
    model = model.cuda()
    model.train()
    # y = y[:,0]

    criterion = nn.BCELoss()
    X = torch.Tensor(X).cuda()
    y = torch.Tensor(y).cuda()
    
    z = X[:,saIndex]
    y[y!=1] = -1
    
    sampler = FairBatch(model, X, y[:,0], z, batch_size = 60, alpha = 0.005, target_fairness = target_fairnes, replacement = False, seed = 20)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.1,
    #     momentum=0.9,
    #     weight_decay=0.01)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    train_data = CustomDataset(X, y, z)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)   

    for epoch in range(100):
        run_epoch(epoch, model, optimizer, criterion, train_loader, train_data) 
    
    model.eval()

    return model
