import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

def data_mod(X, y, batch_size, step_size, input_size, max_time, shuffle=False):
    '''
    This function generates batches of sparse data from the SHD dataset
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

def get_xt(p, step, T, inputs):
    start = p*step
    end = (p+1)*step
    if (end >= T): end=T
        
    x = inputs[ start : end ]
    return x, start, end

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def update_prob_estimates( model, args, train_loader, estimatedDistribution, estimate_class_distribution, first_update=False ):
    PARTS = args.parts
    model.eval()

    print('Find current distribution for each image...')
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        # data = data.view(-1, input_channels, seq_length)
        data = data.to_dense()
        
        B = target.size()[0]
        step = model.network[0].step
        xdata = data.clone()

        T = inputs.size()[0]

        for p in range(PARTS):
            x, start, end = get_xt(p, step, T, inputs)

            with torch.no_grad():
                if p==0:
                    h = model.init_hidden(xdata.size(0))
                else:
                    #_, h = model.network[0].rnn( inputs[:end], h )
                    h = (h[0].detach(), h[1].detach())

                o, h = model.network[0].rnn( x, h )
                out = F.dropout(model.linear2(model.linear1( (h[0]) )), model.dropout)
                out = out.squeeze(dim=0)
                prob_out = F.softmax(out, dim=1)

                if first_update==False:
                    estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p] = prob_out
                else:
                    A = estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p] 
                    B = prob_out
                    estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p] = 0.6*A + 0.4*B
    
    #estimate_class_distribution = torch.zeros(n_classes, PARTS, n_classes, dtype=torch.float)
    
    print('Find best for each class...')
    for batch_idx, (data, target) in enumerate(train_loader):            
        j=0
        for idx in range(batch_idx*batch_size, (batch_idx+1)*batch_size):
            y = target[j].item()
            
            for p in range(PARTS):
                current_distribution = estimatedDistribution[idx, p]
                #print('y=', y, ' --> torch.argmax(current_distribution) ', torch.argmax(current_distribution))
        
                if torch.argmax(current_distribution) == target[j]:
                    #print('estimate_class_distribution[y, p] = ', estimate_class_distribution[y, p].size())
                    #print('current_distribution = ', current_distribution.size())
                    if first_update==False:
                        estimate_class_distribution[y, p] = current_distribution
                    else:
                        if random.randint(0, 8) == 2:
                            estimate_class_distribution[y, p] = current_distribution
                        #estimate_class_distribution[y, p] = 0.9* estimate_class_distribution[y, p] + 0.1*current_distribution
                
            j += 1
            
            
    print('In the current example estimates replace where you have mistakes...')
    for batch_idx, (data, target) in enumerate(train_loader):            
        #if batch_idx%100==0:
        #    print('batch .. ', batch_idx)
        
        j=0
        for idx in range(batch_idx*batch_size, (batch_idx+1)*batch_size):
            y = target[j].item()
            
            for p in range(PARTS):
                current_distribution = estimatedDistribution[idx, p]
                #print('y=', y, ' --> torch.argmax(current_distribution) ', torch.argmax(current_distribution))
        
                if torch.argmax(current_distribution) != target[j]:
                    estimatedDistribution[idx, p] = estimate_class_distribution[y, p] 
                
            j += 1
    
    first_update = True
    return first_update
