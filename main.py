from simulation import update_x, update_xd, update_X, compute_deltaV, compute_VX, Gx,compute_X, compute_x_from_X, compute_e_xd_from_X
from ac import compute_critic_activate, compute_critic, init_critic_weight, compute_actor_activate, compute_actor, \
    init_actor_weight, compute_u, compute_PHI_THETA_C
import numpy as np
import torch as th
import math

from plot import draw_weight, draw_x_xd, draw_value

def step(U, t, x, xd, p):
    next_x = update_x(x, U, p, t)
    next_xd = update_xd(xd)
    # print(x, xd)
    return next_x, next_xd

def update_ac2(History, Wc, Wa, N0):
    Wa_new = []
    Wc_new = []
    for i in range(3):
        first = np.zeros((52, 52))
        second = np.zeros((52, 1))
        for p in range(N0):
            X, U, Xnext, Unext = History[p]
            PHI_p, THETA_p, C_p = compute_PHI_THETA_C(X[i], Xnext[i], U[i], Unext[i], Wc[i], Wa[i])
            first += np.matmul(PHI_p.T, PHI_p)
            second += PHI_p.T * C_p
            # print(C_p)

        # print(first, second)
        new_THETA = np.matmul(np.matrix(first).I, second)
        # print(new_THETA)
        Wc_new.append(new_THETA[:18])
        Wa_new.append(new_THETA[18:])
    # print(Wc_new, Wa_new)
    return Wc_new, Wa_new

def train2(num=50, N0=56):
    record = []
    W_critic = [init_critic_weight() for _ in range(3)]
    W_actor = [init_actor_weight() for _ in range(3)]
    p = [(np.random.rand(3, 1)-0.5) * 2 for _ in range(3)]

    System = {
        'x': [[np.random.rand(2, 1) for _ in range(3)] for _ in range(N0)],
        'xd': [[np.random.rand(2, 1) for _ in range(3)] for _ in range(N0)],
    }

    for i in range(num):
        # print(W_critic)
        history = []
        for m in range(N0):
            X = [compute_X(xi - xdi, xdi) for xi, xdi in zip(System['x'][m], System['xd'][m])]
            U = [compute_u(X[i], W_actor[i])[0, 0] for i in range(3)]
            x_next, xd_next = step(U, i, System['x'][m], System['xd'][m], p)

            X_next = [compute_X(xi - xdi, xdi) for xi, xdi in zip(x_next, xd_next)]
            U_next = [compute_u(X_next[i], W_actor[i])[0,0] for i in range(3)]
            history.append([X, U, X_next, U_next])

            System['x'][m] = x_next
            System['xd'][m] = xd_next

            if p == 0:
                record.append(
                    [W_critic, W_actor,
                     [System['x'][0], System['xd'][0]], U])

        W_critic, W_actor = update_ac2(history, W_critic, W_actor, N0)

    return record


if __name__ == '__main__':
    record = train2(2)
    u_list = [rc[3] for rc in record]
    draw_value(u_list, 'The Control Input')

    Wc_history = [[rc[0][i] for rc in record] for i in range(3)]
    Wa_history = [[rc[1][i] for rc in record] for i in range(3)]

    for index, history in enumerate(Wc_history):
        draw_weight(history, 'The critic NN weight of system ' + str(index))

    for index, history in enumerate(Wa_history):
        draw_weight(history, 'The actor NN weight of system ' + str(index))

    x_xd = [rc[2] for rc in record]
    draw_x_xd(x_xd)

