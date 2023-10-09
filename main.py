import math
import h5py
import argparse
from tqdm import tqdm

import tonic
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from prune import *
from model_module import *

def data_generator(dataset, batch_size, time_slice, datapath, shuffle=True):
    if dataset == 'SHD':
        shd_train = h5py.File(datapath + 'SHD/shd_train.h5', 'r')
        shd_test = h5py.File(datapath + 'SHD/shd_test.h5', 'r')

        shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = batch_size, step_size = time_slice, input_size = 700, max_time = 1.4)
        shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = batch_size, step_size = time_slice, input_size = 700, max_time = 1.4)
        
        # train_loader = shd_train[:int(0.95 * len(shd_train))]
        # val_loader = shd_train[int(0.95 * len(shd_train)):]
        train_loader = shd_train
        test_loader = shd_test
        n_classes = 20
        seq_length = 1.4
        input_channels = 700

    else:
        print('Dataset not included! Use a different dataset.')
        exit(1)
    # return train_loader, val_loader, test_loader, seq_length, input_channels, n_classes
    return train_loader, test_loader, seq_length, input_channels, n_classes

def get_stats_named_params( model ):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone().to(device_1), 0.0*param.detach().clone().to(device_1), 0.0*param.detach().clone().to(device_1)
        named_params[name] = (param.to(device_1), sm, lm, dm)
    return named_params

def post_optimizer_updates( named_params, args, epoch ):
    alpha = args.alpha
    beta = args.beta
    rho = args.rho
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.add_( -alpha * (param - sm) )
        sm.data.mul_( (1.0-beta) )
        sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params, args):
    alpha = args.alpha
    rho = args.rho
    _lambda = args.lmda
    regularization = torch.zeros([]).to(device_1)
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm )
        r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
        regularization += r_p
    return regularization 

def reset_named_params(named_params, args):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.to_dense()

        with torch.no_grad():
            model.eval()

            hidden = model.init_hidden(data.size(0))
            
            outputs, hidden, recon_loss = model(data, hidden) 

            output = outputs[-1]
            test_loss += F.nll_loss(output, target, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= (len(test_loader) * data.shape[0])
    
    return test_loss, 100. * correct / (len(test_loader) * data.shape[0])


def train(epoch, args, train_loader, n_classes, model, named_params, k, progress_bar):
    global estimate_class_distribution

    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    PARTS = args.parts
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_oracle_loss = 0
    model.train()
    
    T = seq_length
    #entropy = EntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.to_dense()
  
        B = target.size()[0]
        # step = model.network.step
        xdata = data.clone()
        pdata = data.clone()
        
        # T = inputs.size()[0]
        
        _PARTS = PARTS
        # if (PARTS * step < T):
        #     _PARTS += 1
        h = model.init_hidden(xdata.size(0))
      
        p_range = range(_PARTS)

        data = torch.split(data, split_size_or_sections=1, dim=1)
        for p in range(len(data)):
            x = data[p]
            if p==p_range[0]:
                h = model.init_hidden(xdata.size(0))
            else:
                h = tuple(v.detach() for v in h)
            
            if p<PARTS-1:
                if epoch <0:
                    oracle_prob = 0*estimate_class_distribution[target, p] + (1.0/n_classes)
                else:
                    oracle_prob = estimate_class_distribution[target, p]
            else:
                oracle_prob = F.one_hot(target, num_classes = 20).float() 

            
            o, h, hs = model.network.forward(x, h ,p)

            prob_out = F.softmax(h[-1], dim=1)
            output = F.log_softmax(h[-1], dim=1) 

            if p<PARTS-1:
                with torch.no_grad():
                    filled_class = [0]*n_classes
                    n_filled = 0
                    for j in range(B):
                        if n_filled==n_classes: break

                        y = target[j].item()
                        if filled_class[y] == 0 and (torch.argmax(prob_out[j]) != target[j]):
                            filled_class[y] = 1
                            estimate_class_distribution[y, p] = prob_out[j].detach()
                            n_filled += 1

            if p%k==0 or p==p_range[-1]:
                optimizer.zero_grad()
                
                nll_loss = F.nll_loss(output, target, reduction='none')
                # clf_loss = ((p+1)/_PARTS) * nll_loss
                # clf_loss = clf_loss.mean()
                clf_loss = ((p+1)/_PARTS) * F.cross_entropy(output, target)
                oracle_loss = (1-((p+1)/_PARTS)) * torch.mean(-oracle_prob * output)
                    
                regularizer = get_regularizer_named_params(named_params, args)
                loss = regularizer + clf_loss + oracle_loss
   
                loss.backward()

                # if args.clip > 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    
                optimizer.step()
                post_optimizer_updates( named_params, args,epoch )

                train_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_regularizaton_loss += regularizer #.item()
                total_oracle_loss += oracle_loss.item()

        progress_bar.update(1)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='SHD', help='Dataset')
parser.add_argument('--datapath', type=str, default= '../data/', help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='Batch size')
parser.add_argument('--parts', type=int, default=20, help='Parts to split the sequential input into')

parser.add_argument('--nlayers', type=int, default=2, help='Number of layers')
parser.add_argument('--nhid', type=int, default=256, help='Number of Hidden units')
parser.add_argument('--epochs', type=int, default=150, help='Number of Epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--when', nargs='+', type=int, default=[20], help='Epochs when Learning rate decays')
parser.add_argument('--optim', type=str, default='Adam', help='Optimiser')
parser.add_argument('--wdecay', type=float, default=0., help='Weight decay')
parser.add_argument('--clip', type=float, default=1., help='Gradient Clipping')
parser.add_argument('--alpha', type=float, default=0.1, help='Weight update parameter (Alpha)')
parser.add_argument('--beta', type=float, default=0.5, help='Weight update parameter (Beta)')
parser.add_argument('--rho', type=float, default=0.0, help='Weight update parameter  (Rho)')
parser.add_argument('--lmda', type=float, default=1.0, help='Regularisation strength (Lambda)')

parser.add_argument('--prun_rate', nargs='+', type=float, default=[0.05, 0.05], help = 'Pruning rate per layer')
parser.add_argument('--reg_rate', nargs='+', type=float, default=[0.05, 0.05], help = 'Regeneration rate per layer')
parser.add_argument('--t_num', type=int, default=5, help = 'Plasticity Threshold')
parser.add_argument('--seed', type=int, default=1111, help='Random seed')

print('PARSING ARGUMENTS...')           
args = parser.parse_args()
args.cuda = True

torch.backends.cudnn.benchmark = True
device_0 = torch.device('cpu')
device_1 = torch.device('cuda:0')
device_2 = torch.device('cuda:1')

# Set the random seed manually for reproducibility.
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.manual_seed(args.seed)

if args.dataset in ['SHD']:
    train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, 
                                                                     batch_size=args.batch_size,
                                                                     time_slice=args.parts,
                                                                     datapath=args.datapath, 
                                                                     shuffle = True)
    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)
    estimatedDistribution = None
else:
    exit(1)

model = SeqModel(ninp = input_channels,
                 nhid = args.nhid,
                 nout = n_classes,
                 n_timesteps = seq_length, 
                 parts = args.parts)

model.cuda()
print('Model: ', model)

optimizer = None
lr = args.lr
all_train_losses, all_test_losses = [], []
all_train_acc, all_test_acc = [], []
epochs = args.epochs
prun_rate2, prun_rate3 = args.prun_rate[0], args.prun_rate[1]
reg_rate2, reg_rate3 = args.reg_rate[0], args.reg_rate[1]
T = args.t_num
START = 10                                       # Pruning starts at this epoch
first_update = False
named_params = get_stats_named_params(model)

if optimizer is None:
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wdecay)
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.2)

# Initialised Weights
curr_w2 = model.network.layer1_x.weight.data.T
curr_w3 = model.network.layer2_x.weight.data.T

# Initialisation of Synaptic Constraint parameters
# Synaptic Boundaries
max_val, max_ind = torch.max(abs(curr_w2), dim=1)
R_pos_2, R_neg_2 = max_val.unsqueeze(1).expand(curr_w2.shape), -max_val.unsqueeze(1).expand(curr_w2.shape)
max_val, max_ind = torch.max(abs(curr_w3), dim=1)
R_pos_3, R_neg_3 = max_val.unsqueeze(1).expand(curr_w3.shape), -max_val.unsqueeze(1).expand(curr_w3.shape)
# Consecutive Time
N_pos_2, N_neg_2, N2 = np.zeros(curr_w2.shape), np.zeros(curr_w2.shape), np.zeros(curr_w2.shape)
N_pos_3, N_neg_3, N3 = np.zeros(curr_w3.shape), np.zeros(curr_w3.shape), np.zeros(curr_w3.shape)
# Accumulated difference
C_pos_2, C_neg_2 =  np.zeros(curr_w2.shape), np.zeros(curr_w2.shape)
C_pos_3, C_neg_3 =  np.zeros(curr_w3.shape), np.zeros(curr_w3.shape)

for epoch in range(1, epochs + 1):  
    if args.dataset in ['SHD']:        
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        
        # Previous Epoch Weights
        prev_w2 = model.network.layer1_x.weight.data.T
        prev_w3 = model.network.layer2_x.weight.data.T

        # Training
        train(epoch, args, train_loader, n_classes, model, named_params, 1, progress_bar)  
        progress_bar.close()

        # Current Epoch Weights
        curr_w2 = model.network.layer1_x.weight.data.T
        curr_w3 = model.network.layer2_x.weight.data.T
        
        reset_named_params(named_params, args)

        # Evaluation
        train_loss, train_acc = test(model, train_loader)
        all_train_losses.append(train_loss)
        all_train_acc.append(train_acc)
        print('\nLoss:', train_loss, end = '\t')
        print('Accuracy:', float(train_acc.item()))

        test_loss, test_acc = test(model, test_loader)
        all_test_losses.append(test_loss)
        all_test_acc.append(test_acc)
        print('Test Loss:', test_loss, end = '\t')
        print('Test Accuracy:', float(test_acc.item()))
        
        # if epoch%5 == 0:
            # val_loss, val_acc = test(model, val_loader)
            # print('Validation Loss:', val_loss, end = '\t')
            # print('Validation Accuracy:', val_acc.item())
            # test_loss, test_acc = test(model, test_loader)
            # all_test_losses.append(test_loss)
            # all_test_acc.append(test_acc)
            # print('Test Loss:', test_loss, end = '\t')
            # print('Test Accuracy:', test_acc.item())

        # print(curr_w2[:6])
        curr_w2, R_pos_2, R_neg_2, C_pos_2, C_neg_2, N_pos_2, N_neg_2, N2 = synaptic_constraint(curr_w2, prev_w2, R_pos_2, R_neg_2, C_pos_2, C_neg_2, N_pos_2, N_neg_2, N2, T)
        # print(curr_w2[:6])
        curr_w3, R_pos_3, R_neg_3, C_pos_3, C_neg_3, N_pos_3, N_neg_3, N3 = synaptic_constraint(curr_w3, prev_w3, R_pos_3, R_neg_3, C_pos_3, C_neg_3, N_pos_3, N_neg_3, N3, T)
        curr_w4 = model.network.layer3_x.weight.data.T

        if epoch > START:
            curr_w2, prun_rate2, reg_rate2 = plasticity(curr_w2, curr_w3, R_pos_2, R_neg_2, prun_rate2, reg_rate2, T, model, 'h1', epoch)
            model.network.layer1_x.weight.data = curr_w2.T
            curr_w3, prun_rate3, reg_rate3 = plasticity(curr_w3, curr_w4, R_pos_3, R_neg_3, prun_rate3, reg_rate3, T, model, 'h2', epoch)
            model.network.layer2_x.weight.data = curr_w3.T
            
        # if epoch in args.when :
        #     lr *= 0.1
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

plot_info(all_train_losses, all_test_losses, 'loss', args)
plot_info(all_train_acc, all_test_acc, 'acc', args)
