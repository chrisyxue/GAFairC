import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import types
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm


from pytorch_transformers import AdamW, WarmupLinearSchedule



class Group_DRO_Loss:
    def __init__(self, criterion,is_robust, X, y, params, alpha=0.2, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False):
        # params contain saIndex, saValue
        
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.criterion = criterion

        self.n_groups = 4
        self.saIndex = params['saIndex']
        self.saValue = params['saValue']
        y_G_p = y[X[:,self.saIndex] == self.saValue]
        y_G_p_pos = y_G_p[y_G_p == 1]
        y_G_p_neg = y_G_p[y_G_p == -1]
        y_G_np = y[X[:,self.saIndex] != self.saValue]
        y_G_np_pos = y_G_np[y_G_np == 1]
        y_G_np_neg = y_G_np[y_G_np == -1]

        self.group_counts = torch.FloatTensor([y_G_p_pos.shape[0],y_G_p_neg.shape[0],y_G_np_pos.shape[0],y_G_np_neg.shape[0]])
        self.group_frac = self.group_counts/self.group_counts.sum()

        # self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()

def run_epoch(epoch, model, optimizer, X, y, g, loss_computer,scheduler=None,batch_size=36):
    """
    scheduler is only used inside this function if model is bert.
    """

    model.train()

    num_data = X.shape[0]
    max_idx = int(num_data/batch_size) - 1
    y[y == -1] = 0

    for batch_idx in range(max_idx):
        x_batch = X[batch_idx*batch_size:(batch_idx+1)*batch_size]
        y_batch = y[batch_idx*batch_size:(batch_idx+1)*batch_size]
        g_batch = g[batch_idx*batch_size:(batch_idx+1)*batch_size]

        x_batch = torch.Tensor(x_batch).cuda()
        y_batch = torch.Tensor(y_batch).cuda()
        y_batch = y_batch.long()
        g_batch = torch.Tensor(g_batch).cuda()

        y_batch = torch.squeeze(y_batch)
        g_batch = torch.squeeze(g_batch)

        outputs = model(x_batch)
        
        loss_main = loss_computer.loss(outputs, y_batch, g_batch, is_training=True)

        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()
    
    loss_computer.reset_stats()

class DRO_Module(nn.Module):
    def __init__(self,in_dim):
        super(DRO_Module,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim*2),
            nn.ReLU(),
            nn.Linear(in_dim*2,2)
        )

    def forward(self, x):
        x=self.net(x)
        return x


def train(X,y,saIndex,saValue):

    model = DRO_Module(in_dim=X.shape[-1])
    model = model.cuda()

    criterion = nn.CrossEntropyLoss(reduction='none')
    
    params = {}
    params['saIndex'] = saIndex
    params['saValue'] = saValue

    print(params)

    g = np.zeros_like(y)
    for i in range(g.shape[0]):
        if X[i,saIndex] == saValue and y[i] == 1:
            g[i] = 0
        elif X[i,saIndex] == saValue and y[i] != 1:
            g[i] = 1
        elif X[i,saIndex] != saValue and y[i] == 1:
            g[i] = 2
        else:
            g[i] = 3


    # optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.01)
    scheduler = None

    # process generalization adjustment stuff
    adjustments = [float(c) for c in str(0).split(',')]
    assert len(adjustments) in (1, 4)
    if len(adjustments)==1:
        adjustments = np.array(adjustments*4)
    else:
        adjustments = np.array(adjustments)

    # initial loss computer
    train_loss_computer = Group_DRO_Loss(criterion,True,X,y,params)

    for epoch in range(100):
        run_epoch(epoch, model, optimizer, X, y, g, train_loss_computer, scheduler=None, batch_size=36)
    
    model.eval()
        
    return model
