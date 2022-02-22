import random

from simulation import update_x, update_xd, update_X, compute_deltaV, compute_VX, Gx,compute_X, compute_x_from_X, compute_e_xd_from_X
from ac import compute_critic_activate, compute_critic, init_critic_weight, compute_actor_activate, compute_actor, \
    init_actor_weight, compute_u, compute_PHI_THETA_C
import numpy as np

from constant import Constant

from plot import draw_weight, draw_x_xd, draw_value

def step(U, t, x, xd, p):
    next_x = update_x(x, U, p, t)
    next_xd = update_xd(xd)
    # print(x, xd)
    return next_x, next_xd

def update_ac2(History, Wc, Wa):
    Wa_new = []
    Wc_new = []
    for i in range(3):
        first = np.zeros((52, 52))
        second = np.zeros((52, 1))
        count = 0
        for p in range(len(History)):
            X, U, Xnext, Unext = History[p]
            try:
                PHI_p, THETA_p, C_p = compute_PHI_THETA_C(X[i], Xnext[i], U[i], Unext[i], Wc[i], Wa[i])
            except:
                continue
            # print(PHI_p.T.shape, C_p.shape)
            if np.isnan(PHI_p).any() or np.isnan(THETA_p).any() or np.isnan(C_p).any():
                continue

            count += 1
            first += np.matmul(PHI_p.T, PHI_p)
            second += PHI_p.T * C_p

        print(i, 'real_N0', count)
        try:
            new_THETA = np.matmul(np.matrix(first).I, second)
        except:
            Wc_new.append(Wc[i])
            Wa_new.append(Wa[i])
            print(i, 'fail compute theta')
            continue

        # print(new_THETA)
        Wc_new.append(new_THETA[:18])
        Wa_new.append(new_THETA[18:])
    # print(Wc_new, Wa_new)
    return Wc_new, Wa_new

def train2(n_round=625, n_step = 5, n_train=20, n_test=21):
    W_critic = [init_critic_weight() for _ in range(3)]
    W_actor = [init_actor_weight() for _ in range(3)]
    p = [(np.random.rand(3, 1)-0.5) * 2 for _ in range(3)]


    w_record = []
    w_record.append([W_critic, W_actor])
    for iter in range(n_train):
        print(iter, 'sample and train')

        history = []
        for i in range(n_round):
            x = [np.random.rand(2, 1) for _ in range(3)]
            xd = [np.random.rand(2, 1) for _ in range(3)]
            for t in range(n_step):
                X = [compute_X(xi - xdi, xdi) for xi, xdi in zip(x, xd)]
                U = [compute_u(X[i], W_actor[i])[0, 0] for i in range(3)]

                x_next, xd_next = step(U, t, x, xd, p)
                # print(x_next)
                X_next = [compute_X(xi - xdi, xdi) for xi, xdi in zip(x_next, xd_next)]
                U_next = [compute_u(X_next[i], W_actor[i])[0, 0] for i in range(3)]
                history.append([X, U, X_next, U_next])

                x = x_next
                xd = xd_next


        W_critic, W_actor = update_ac2(history, W_critic, W_actor)
        w_record.append([W_critic, W_actor])

    print('test')

    test_record = []
    x = [np.random.rand(2, 1) for _ in range(3)]
    xd = [np.random.rand(2, 1) for _ in range(3)]
    for t in range(n_test):

        X = [compute_X(xi - xdi, xdi) for xi, xdi in zip(x, xd)]
        U = [compute_u(X[i], W_actor[i])[0, 0] for i in range(3)]

        test_record.append([(x, xd), U])
        x_next, xd_next = step(U, t, x, xd, p)
        # print(x_next)
        x = x_next
        xd = xd_next

    return w_record, test_record


if __name__ == '__main__':
    w_record, test_record = train2()

    Wc_history = [[rc[0][i] for rc in w_record] for i in range(3)]
    Wa_history = [[rc[1][i] for rc in w_record] for i in range(3)]

    for index, history in enumerate(Wc_history):
        draw_weight(history, 'The critic NN weight of system ' + str(index))

    for index, history in enumerate(Wa_history):
        draw_weight(history, 'The actor NN weight of system ' + str(index))

    x_xd = [rc[0] for rc in test_record]
    draw_x_xd(x_xd)

    u_list = [rc[1] for rc in test_record]
    draw_value(u_list, 'The Control Input')

