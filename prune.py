import numpy as np
import torch
import torch.nn as nn
import numpy as np

def synaptic_constraint(curr_w, prev_w, R_pos, R_neg, C_pos, C_neg, N_pos, N_neg, N, T):
    '''
    Function that caculates the synaptic boundaries for a layer, given the current and previous epoch weights, and Plasticity Theshold
    INPUT: curr_w (Tensor), prev_w (Tensor), T (Tensor)
    OUTPUT: curr_w (Tensor), R_pos (Tensor), R_neg (Tensor)
    '''
    E = 0.75                                        # Boundary Shrinking Factor
    T = np.full(curr_w.shape, T)                    # Plasticity Threshold

    # Constraining synapses and calculating synaptic activity
    for i in range(curr_w.shape[0]):
        for j in range(curr_w.shape[1]):
            if curr_w[i][j] > R_pos[i][j]:
                N_pos[i][j] += 1
                C_pos[i][j] += curr_w[i][j] - R_pos[i][j]
                curr_w[i][j] = R_pos[i][j]
            else:
                N_pos[i][j], C_pos[i][j] = 0, 0

            if curr_w[i][j] < R_neg[i][j]:
                N_neg[i][j] += 1
                C_neg[i][j] += R_neg[i][j] - curr_w[i][j]
                curr_w[i][j] = R_neg[i][j]
            else:
                N_neg[i][j], C_neg[i][j] = 0, 0

            if curr_w[i][j] < prev_w[i][j]: N[i][j] += 1
            else: N[i][j] = 0

            # Updating Synaptic Boundary
            if N_pos[i][j] > T[i, j]:
                R_pos[i][j] += C_pos[i][j] / T[i, j]
            if N_neg[i][j] > T[i, j]:
                R_neg[i][j] -= C_neg[i][j] / T[i, j]
            if N[i][j] > T[i, j]:
                R_pos[i][j], R_neg[i][j] = E * R_pos[i][j], E * R_neg[i][j]

    return curr_w, R_pos, R_neg, C_pos, C_neg, N_pos, N_neg, N

def plasticity(clw, nlw, R_pos, R_neg, prun_rate, reg_rate, T, model, layer, epoch):
    '''
    Function that prunes and generates the connections between the presynaptic and postsynaptic neurons, given the current and next layer weights, synaptic boundaries, pruning rate, regeneration rate, and plasticity threshold 
    INPUT: clw (Tensor), plw (Tensor), R_pos (Tensor), R_neg (Tensor), prun_rate (float), reg_rate (float), T (int), model (nn.module), layer (string)
    OUTPUT: clw (Tensor), prun_rate (float), reg_rate (float)
    '''
    prun_a, prun_b = 1, 0.00075                       # Pruning constants for updates
    reg_g = 1.1                                       # Regeneration constant for updates
    T_num = np.full(clw.shape, T)                     # Plasticity Threshold
    START, MID = 5, 60                                # Pruning starts and slows at these epoch

    #------------------------------------ Pruning ---------------------------------------#
    R_range = R_pos - R_neg                           # Range of the synaptic boundaries
    D = torch.sum(R_range, dim=0)                     # Activity Level

    # Pruning neurons based on D
    no_prun_neu = round(256 * prun_rate)
    indices = torch.argsort(D, dim=0)[:no_prun_neu]
    for i in indices:
        clw[:, i] = 0

    # Updating pruning rate
    if epoch <= MID:    d = prun_a * np.exp(-(epoch - START))
    else:               d = prun_b

    no_neu = torch.all(clw != 0, dim=0)
    N_cl = no_neu.sum().item()
    if layer == 'h2':
         no_neu = torch.all(nlw != 0, dim=0)
         N_nl = no_neu.sum().item()
    else:
         N_nl = 20

    prun_rate += (d * N_cl/N_nl)

    #---------------------------------- Regeneration ------------------------------------#
    for name, param in model.named_parameters():
        if name == 'weight' and param.requires_grad:
            dL = param.grad
            dL = dL.T
            no_syn_reg = round(dL.shape[0] * dL.shape[1] * reg_rate)
            T_g = torch.zeros(dL.shape)

            # Regeneration update
            for i in range(T_g.shape[0]):
                for j in range(T_g.shape[1]):
                    if j in indices:
                        T_g[i, j] += 1
                    else: T_g[i, j] = 0

            # Condition that checks if no of connections that can be regenerated is greater than the regeneration rate allowed
            no_syn = torch.count_nonzero(T_g).item()
            if no_syn > no_syn_reg:
                topk_values, topk_indices = torch.topk(T_g.view(-1), k=no_syn_reg)
            else:
                topk_values, topk_indices = torch.topk(T_g.view(-1), k=no_syn)

            # Regenerating synapases
            r = topk_indices // T_g.shape[1]
            c = topk_indices % T_g.shape[1]
            for i, j in zip(r, c):
                if T_g[i, j] > T_num[i, j]:
                    clw[i, j] = clw[i, j] - (model.l_r * dL[i, j])

            # Updating regeneration rate
            reg_rate += np.power(reg_g, epoch - START)

    if layer == 'h1':
        ldim = torch.all(clw != 0, dim=0)
        ldim = ldim.sum().item()
        print('NUmber of neurons in Layer 2: ', ldim)
    elif layer == 'h2':
        ldim = torch.all(clw != 0, dim=0)
        ldim = ldim.sum().item()       
        print('Number of neurons in Layer 3: ', ldim)

    return clw, prun_rate, reg_rate
