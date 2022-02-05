from simulation import update_x, update_xd, update_X, compute_deltaV, compute_VX, Gx,compute_X, compute_x_from_X, compute_e_xd_from_X
from ac import compute_critic_activate, compute_critic, init_critic_weight, compute_actor_activate, compute_actor, \
    init_actor_weight, compute_u, compute_PHI_THETA_C
import numpy as np
import torch as th
import math

from plot import draw_weight, draw_x_xd, draw_value

def step(X, U, t, x, xd, p):
    next_x = update_x(x, U, p, t)
    next_xd = update_xd(xd)
    # print(x, xd)
    next_X = [compute_X(xi - xdi, xdi) for xi, xdi in zip(x, xd)]

    X1, X2, X3 = X
    u1, u2, u3 = U
    V1 = compute_VX(X1, u1, t)
    V2 = compute_VX(X2, u2, t)
    V3 = compute_VX(X3, u3, t)
    return next_X, [V1, V2, V3], next_x, next_xd


def update_ac(X, X_next, V, U, Wc, Wa, learning_rate=0.1):
    Gnext = Gx(X_next)
    C_loss = []
    A_loss = []
    Wc_new = []
    Wa_new = []
    for i in range(3):
        tensorX = th.tensor(compute_critic_activate(X[i]))
        tensorW = th.tensor(Wc[i], requires_grad=True)
        Vi_predict = th.matmul(tensorX, tensorW)[0, 0]
        critic_loss = (V[i] - Vi_predict) ** 2
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(tensorW, 100.0)

        tensorW_ = tensorW - learning_rate * tensorW.grad
        Wc_new.append(tensorW_.data.numpy())

        R = 1
        alpha = 3
        deltaV = compute_deltaV(X[i], X_next[i], U[i], V[i]).T
        tensorXa = th.tensor(compute_actor_activate(X[i]))
        tensorWa = th.tensor(Wa[i], requires_grad=True)
        pi_pridict = alpha * th.tanh(th.matmul(tensorXa, tensorWa)[0, 0])

        critic_new = compute_critic(X_next[i], Wc_new[i])[0][0]
        # print(i, critic_new, Vi_predict, pi_pridict)

        pi_target = th.FloatTensor(- alpha * np.tanh((1 / 2 * alpha) * (1 / R) * Gnext[i] * deltaV))
        # pi_target = - alpha * np.tanh((1 / 2 * alpha) * (1 / R) * Gnext[i] * (V[i] - Vi_predict.detach().numpy()))
        # pi_target = - alpha * np.tanh((1 / 2 * alpha) * (1 / R) * Gnext[i] * (critic_new - V[i]))

        # print(i, deltaV, pi_target, pi_pridict)
        # actor_loss = (pi_pridict - pi_target) ** 2
        actor_loss = ((pi_pridict - pi_target) ** 2).sum()


        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(tensorWa, 100.0)

        tensorWa_ = tensorWa - learning_rate * tensorWa.grad
        Wa_new.append(tensorWa_.data.numpy())
        C_loss.append(critic_loss.data)
        A_loss.append(actor_loss.data)

    return Wc_new, Wa_new, C_loss, C_loss

def train(num=50, learning_rate=0.01):
    record = []

    W_critic = [init_critic_weight() for _ in range(3)]
    W_actor = [init_actor_weight() for _ in range(3)]
    # X = [np.random.rand(4, 1) for _ in range(3)]
    # X = [np.zeros((4, 1)) for _ in range(3)]
    # x = [np.zeros((2, 1)) for _ in range(3)]
    x = [np.random.rand(2, 1) for _ in range(3)]
    # xd = [np.zeros((2, 1)) for _ in range(3)]
    xd = [np.random.rand(2, 1) for _ in range(3)]
    p = [(np.random.rand(3, 1)-0.5) * 2 for _ in range(3)]
    X = [compute_X(xi - xdi, xdi) for xi, xdi in zip(x, xd)]

    record.append([X, W_critic, W_actor, [0,0,0], [0,0,0], [0,0,0], [x, xd], [0, 0, 0]])

    for i in range(num):
        U = [compute_u(X[i], W_actor[i])[0,0] for i in range(3)]
        print('step ', i)
        print('x ', x)
        print('xd ', xd)
        print('U ', U)
        X_next, V, x, xd = step(X, U, i, x, xd, p)
        # print(len(X), X[1].size, len(X_next), X_next[1].size)
        W_critic, W_actor, critic_loss, actor_loss = update_ac(X, X_next, V, U, W_critic, W_actor, learning_rate)

        print('critic_loss ', critic_loss)
        print('actor_loss ', actor_loss)

        X = X_next
        record.append([X, W_critic, W_actor, V, critic_loss, actor_loss, [x, xd], U])

    return record


if __name__ == '__main__':
    record = train(10)
    V = [rc[3] for rc in record]
    draw_value(V, 'The Values')

    c_loss = [rc[4] for rc in record]
    draw_value(c_loss, 'The critic losses of systems')

    a_loss = [rc[5] for rc in record]
    draw_value(a_loss, 'The actor losses of systems')

    u_list = [rc[7] for rc in record]
    draw_value(u_list, 'The Control Input')

    Wc_history = [[rc[1][i] for rc in record] for i in range(3)]
    Wa_history = [[rc[2][i] for rc in record] for i in range(3)]

    for index, history in enumerate(Wc_history):
        draw_weight(history, 'The critic NN weight of system ' + str(index))

    for index, history in enumerate(Wa_history):
        draw_weight(history, 'The actor NN weight of system ' + str(index))

    x_xd = [rc[6] for rc in record]
    draw_x_xd(x_xd)

