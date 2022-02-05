import numpy as np
from scipy import integrate
from constant import Constant


def compute_critic_activate(Xi):
    Xi1, Xi2, Xi3, Xi4 = Xi
    phi = np.array([Xi1 ** 2,
                    Xi1 * Xi2,
                    Xi1 * Xi3,
                    Xi1 * Xi4,
                    Xi2 ** 2,
                    Xi2 * Xi3,
                    Xi2 * Xi4,
                    Xi3 ** 2,
                    Xi3 * Xi4,
                    Xi4 ** 2,
                    Xi1 ** 3,
                    Xi2 ** 3,
                    Xi3 ** 3,
                    Xi4 ** 3,
                    Xi1 ** 4,
                    Xi2 ** 4,
                    Xi3 ** 4,
                    Xi4 ** 4]).T
    return phi


def init_critic_weight():
    return np.zeros((18, 1))


def compute_actor_activate(Xi):
    Xi1, Xi2, Xi3, Xi4 = Xi
    psi = np.array([Xi1,
                    Xi2,
                    Xi3,
                    Xi4,
                    Xi1 ** 2,
                    Xi1 * Xi2,
                    Xi1 * Xi3,
                    Xi1 * Xi4,
                    Xi2 ** 2,
                    Xi2 * Xi3,
                    Xi2 * Xi4,
                    Xi3 ** 2,
                    Xi3 * Xi4,
                    Xi4 ** 2,
                    Xi1 ** 3,
                    Xi1 * Xi2 * Xi3,
                    Xi1 * Xi2 * Xi4,
                    Xi1 * Xi3 * Xi4,
                    Xi1 ** 2 * Xi2,
                    Xi1 ** 2 * Xi3,
                    Xi1 ** 2 * Xi4,
                    Xi2 ** 3,
                    Xi2 ** 2 * Xi1,
                    Xi2 ** 2 * Xi3,
                    Xi2 ** 2 * Xi4,
                    Xi2 * Xi3 * Xi4,
                    Xi3 ** 3,
                    Xi3 ** 2 * Xi1,
                    Xi3 ** 2 * Xi2,
                    Xi3 ** 2 * Xi4,
                    Xi4 ** 3,
                    Xi4 ** 2 * Xi1,
                    Xi4 ** 2 * Xi2,
                    Xi4 ** 2 * Xi3]).T
    return psi

def init_actor_weight():
    return np.ones((34, 1)) * (-0.5)

def compute_critic(Xi, w):
    phi = compute_critic_activate(Xi)
    return np.matmul(phi, w)

def compute_actor(Xi, w):
    psi = compute_actor_activate(Xi)
    return np.matmul(psi, w)

def compute_u(Xi, w):
    mu = compute_actor(Xi, w)
    return Constant.alpha * np.tanh(mu)


def compute_A(Xi, Xinext):
    return (1 + Constant.lamb / 2) * compute_critic_activate(Xi) + (-1 + Constant.lamb / 2) * compute_critic_activate(Xinext)

def compute_B(j, ui, uinext, Wa, psi, psinext):
    def f(uf, psif):
        psij = psif[:, j:j+1]
        return (Constant.alpha * np.tanh(Wa[j] * psij) - uf) * psij
    return 2 * Constant.alpha * Constant.R * (f(ui, psi) + f(uinext, psinext)) / 2

def compute_C(Xi, Xinext, Wa):
    def f1(Xf):
        return np.matmul(np.matmul(Xf.T, Constant.Q), Xf)
    def f2(Xf):
        return 2 * (Constant.alpha ** 2) * compute_actor(Xf, Wa).T * Constant.R * np.tanh(compute_actor(Xf, Wa))
    def f3(Xf):
        # l = list([np.log(1 - np.tanh(Wa[j] * compute_actor_activate(Xf)[:, j]) ** 2) for j in range(len(Wa))])
        l = list([min(100.0, max(-100.0, np.log(1 - np.tanh(Wa[j] * compute_actor_activate(Xf)[:, j]) ** 2))) for j in range(len(Wa))])
        return ((Constant.alpha ** 2) * Constant.R
                * sum(l))
    return (f1(Xi) + f1(Xinext)) / 2 + (f2(Xi) + f2(Xinext)) / 2 + (f3(Xi) + f3(Xinext)) / 2


def compute_PHI_THETA_C(Xi, Xinext, ui, uinext, Wc, Wa):
    PHI = [compute_A(Xi, Xinext)]
    psi = compute_actor_activate(Xi)
    psinext = compute_actor_activate(Xinext)
    for j in range(len(Wa)):
        PHI.append(compute_B(j, ui, uinext, Wa, psi, psinext))
    THETA = np.concatenate([Wc, Wa])
    PHI = np.concatenate(PHI, axis=-1)
    C = compute_C(Xi, Xinext, Wa)
    # print(THETA.shape, PHI.shape, C.shape)
    return PHI, THETA, C
