import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F

def data_mod(X, y, batch_size, step_size, input_size, max_time, shuffle=False):
    '''
    Modifies the SHD dataset into sparse data and batches them.
    ARGS: X (tensor), y (tensor), batch_size (int), step_size (int), input_size (int), max_time (float), shuffle (bool)
    RETURNS: mod_data (list(tuple(tensors)))
    '''
    labels = np.array(y, int)
    nb_batches = len(labels)//batch_size
    sample_index = np.arange(len(labels))

    firing_times = X['times']
    units_fired = X['units']

    time_bins = np.linspace(0, max_time, num=step_size)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    mod_data = []
    while counter<nb_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[2].extend(units)
            coo[1].extend(times)

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,step_size,input_size])).to('cuda:0')
        y_batch = torch.tensor(labels[batch_index]).to('cuda:0')
        
        mod_data.append((X_batch, y_batch))

        counter += 1

    return mod_data

# class EntropyLoss(nn.Module):
#     def __init__(self):
#         super(EntropyLoss, self).__init__()

#     def forward(self, x):
#         b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
#         b = -1.0 * b.sum()
#         return b

def plot_info(tr, te, type, args):
    if type == 'loss':
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, args.epochs+1), tr, label='Training', color='red')
        plt.plot(range(1, args.epochs+1), te, label='Testing', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.savefig('dyn_plots/parts_' + str(args.parts) + '_nhid' + str(args.nhid) + '_loss_plot.png')

    elif type == 'acc':
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, args.epochs+1), tr, label='Training', color='red')
        plt.plot(range(1, args.epochs+1), te, label='Testing', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.savefig('dyn_plots/parts_' + str(args.parts) + '_nhid' + str(args.nhid) + '_acc_plot.png')

    elif type == 'syns':
        prun_ = [i[0] for i in tr]
        reg_ = [i[1] for i in tr]
        tot_ = [i[2] for i in tr]
        
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, args.epochs+1), prun_, label='Pruned', color='red')
        plt.plot(range(1, args.epochs+1), reg_, label='Regenerated', color='green')
        plt.plot(range(1, args.epochs+1), tot_, label='Total', color='black')
        plt.xlabel('Epoch')
        plt.ylabel('Connections/Synapses')
        plt.title('Number of Connections of Hidden Layer 1 Over Epochs')
        plt.legend()
        plt.savefig('dyn_plots/h1_prun_rate_' + str(args.prun_rate[0]) + '_reg_rate_' + str(args.reg_rate[0]) + '.png')

        prun_ = [i[0] for i in te]
        reg_ = [i[1] for i in te]
        tot_ = [i[2] for i in te]

        plt.figure(figsize=(12, 5))
        plt.plot(range(1, args.epochs+1), prun_, label='Pruned', color='red')
        plt.plot(range(1, args.epochs+1), reg_, label='Regenerated', color='green')
        plt.plot(range(1, args.epochs+1), tot_, label='Total', color='black')
        plt.xlabel('Epoch')
        plt.ylabel('Connections/Synapses')
        plt.title('Number of Connections of Hidden Layer 2 Over Epochs')
        plt.legend()
        plt.savefig('dyn_plots/h2_prun_rate_' + str(args.prun_rate[1]) + '_reg_rate_' + str(args.reg_rate[1]) + '.png')
        
    elif type == 'plasticity2':
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, args.epochs+1), tr, label='Pruning rate', color='red')
        plt.plot(range(1, args.epochs+1), te, label='Regeneration rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Plasticity')
        plt.title('Plasticity Over Epochs')
        plt.legend()
        plt.savefig('dyn_plots/layer2_plasticity_pr_rr.png')

    elif type == 'plasticity3':
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, args.epochs+1), tr, label='Pruning rate', color='red')
        plt.plot(range(1, args.epochs+1), te, label='Regeneration rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Plasticity')
        plt.title('Plasticity Over Epochs')
        plt.legend()
        plt.savefig('dyn_plots/layer3_plasticity_pr_rr.png')
