import numpy as np
import torch
import torch.nn as nn


def synaptic_constraint(curr_w, prev_w):
  '''
  # INSERT DOCSTRING IN EVERY FUNCTION FOIR STYLE POINTS
  '''
    # Synaptic Boundaries
    max_val, max_ind = torch.max(abs(curr_w), dim=1)
    R_pos, R_neg = max_val.unsqueeze(1).expand(curr_w.shape), -max_val.unsqueeze(1).expand(curr_w.shape)
    # Consecutive Time
    N_pos, N_neg, N = np.zeros(curr_w.shape), np.zeros(curr_w.shape), np.zeros(curr_w.shape)
    # Accumulated difference
    C_pos, C_neg =  np.zeros(curr_w.shape), np.zeros(curr_w.shape)

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

    return curr_w, R_pos, R_neg


def plasticity(clw, nlw, R_pos, R_neg, prun_rate, reg_rate, T_num, model, layer):
  '''
  # INSERT DOCSTRING IN EVERY FUNCTION FOIR STYLE POINTS
  '''
    prun_a, prun_b = 1, 0.00075                       # Pruning constants for updates
    reg_g = 1.1                                       # Regeneration constant for updates

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
    if layer == 'hl':
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

  return clw, prun_rate, reg_rate
