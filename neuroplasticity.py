import torch
import torch.nn as nn

def synaptic_constraint(curr_w, prev_w, C_pos, C_neg, N_pos, N_neg, N, T):
    '''
    Function that caculates the synaptic boundaries for a layer, given the current and previous epoch weights, and Plasticity Theshold
    INPUT: curr_w (Tensor), prev_w (Tensor), T (Tensor)
    OUTPUT: curr_w (Tensor), R_pos (Tensor), R_neg (Tensor)
    '''
    E = 0.75                                        # Boundary Shrinking Factor
    T = torch.full(curr_w.shape, T)                 # Plasticity Threshold

    # Initialisation of Synaptic Constraint parameters
    # Synaptic Boundaries
    max_val, max_ind = torch.max(abs(curr_w), dim=1)
    R_pos, R_neg = max_val.unsqueeze(1).expand(curr_w.shape), -max_val.unsqueeze(1).expand(curr_w.shape)

    # Constraining synapses and calculating synaptic activity
    pos_mask = curr_w > R_pos
    neg_mask = curr_w < R_neg
    w_mask = curr_w < prev_w

    ## Positive updates
    N_pos[pos_mask] += 1
    C_pos[pos_mask] += curr_w[pos_mask] - R_pos[pos_mask]
    curr_w[pos_mask] = R_pos[pos_mask]
    N_pos[~pos_mask], C_pos[~pos_mask] = 0, 0

    ## Negative updates
    N_neg[neg_mask] += 1
    C_neg[neg_mask] += R_neg[neg_mask] - curr_w[neg_mask]
    curr_w[neg_mask] = R_neg[neg_mask]
    N_neg[~neg_mask], C_neg[~neg_mask] = 0, 0

    N[w_mask] += 1
    N[~w_mask] = 0

    # Updating Synaptic Boundary
    temp_pos_mask = N_pos > T
    temp_neg_mask = N_neg > T
    temp_mask = N > T

    # To avoid warnings
    if R_pos.is_shared() and R_neg.is_shared():
        R_pos = R_pos.clone()
        R_neg = R_neg.clone()
        
    R_pos[temp_pos_mask] += C_pos[temp_pos_mask] / T[temp_pos_mask]
    R_neg[temp_neg_mask] -= C_neg[temp_neg_mask] / T[temp_neg_mask]
    R_pos[temp_mask], R_neg[temp_mask] = E * R_pos[temp_mask], E * R_neg[temp_mask]

    return curr_w, R_pos, R_neg, C_pos, C_neg, N_pos, N_neg, N

def plasticity(clw, nlw, R_pos, R_neg, prun_rate, reg_rate, T, T_g, model, layer, lr, epoch):
    '''
    Function that prunes and generates the connections between the presynaptic and postsynaptic neurons, given the current and next layer weights, synaptic boundaries, pruning rate, regeneration rate, and plasticity threshold 
    INPUT: clw (Tensor), plw (Tensor), R_pos (Tensor), R_neg (Tensor), prun_rate (float), reg_rate (float), T (int), model (nn.module), layer (string)
    OUTPUT: clw (Tensor), prun_rate (float), reg_rate (float)
    '''
    prun_a, prun_b = 1, 0.00075                       # Pruning constants for updates
    reg_g = 1.1                                       # Regeneration constant for updates
    T_num = torch.full(clw.shape, T)                  # Plasticity Threshold
    START, MID = 20, 50                               # Pruning starts and slows at these epoch

    #------------------------------------ Pruning ---------------------------------------#
    R_range = R_pos - R_neg                           # Range of the synaptic boundaries
    D = torch.sum(R_range, dim=0)                     # Activity Level

    # Pruning neurons based on D
    no_prev_conns = torch.count_nonzero(clw).item()
    factor_ = 700 if layer == 'h1' else 256 if layer == 'h2' else None
    no_prun_neu = int((no_prev_conns / factor_) * prun_rate)

    if no_prun_neu == 0:
        no_prun_conn = 0
        print('Number of connections pruned in {0} Layer: '.format(layer), 0)
    else:
        vals_, indices = torch.topk(D, no_prun_neu, largest=False)
        clw[:, indices] = 0
    
        no_prun_conn = no_prev_conns - torch.count_nonzero(clw).item()
        print('Number of connections pruned in {0} Layer: '.format(layer), no_prun_conn)
        
    # Updating pruning rate
    if epoch <= MID:    d = prun_a * torch.exp(-torch.Tensor([epoch - START]))
    else:               d = prun_b

    ncl = torch.count_nonzero(clw).item() / factor_
    nnl =  (torch.count_nonzero(nlw).item() / 256) if layer == 'h1' else 20 if layer == 'h2' else None
    prun_rate += (d * ncl / nnl)
    if torch.is_tensor(prun_rate): prun_rate = prun_rate.item()
    while prun_rate > 0.05: prun_rate = 0.01

    #---------------------------------- Regeneration ------------------------------------#
    for name, param in model.named_parameters():
        if param.requires_grad:
            prun_layer = False
            layer_name = None

            if (layer == 'h1') and ('1_x.weight' in name):
                prun_layer = True
                layer_name = 'h1'
            elif (layer == 'h2') and ('2_x.weight' in name):
                prun_layer = True
                layer_name = 'h2'
            
            if prun_layer:
                dL = param.grad.T
                no_syn_reg = round(dL.shape[0] * dL.shape[1] * reg_rate)
    
                vals_, indices = torch.topk(dL.reshape(-1), no_syn_reg, largest=True)
                r, c = indices // dL.shape[1], indices % dL.shape[1]
                T_g[r, c] += 1
                mask = torch.zeros_like(T_g, dtype=torch.bool)
                mask[r, c] = clw[r, c] != 0  
                T_g[mask] = 0
    
                conn_mask = T_g > T_num
                clw[conn_mask] -= lr * dL[conn_mask]
                T_g[conn_mask] = 0
                reg_count = conn_mask.sum().item() 
                print('Connections regenerated in {0} Layer: '.format(layer_name), reg_count)
        
    # Updating regeneration rate
    reg_rate += pow(reg_g, epoch - START)
    while reg_rate > 0.99: reg_rate = 0.99

    no_syns = torch.count_nonzero(clw).item()
    print('Total connections in {0} Layer: '.format(layer), no_syns)

    return clw, prun_rate, reg_rate, T_g, [no_prun_conn, reg_count, no_syns]
