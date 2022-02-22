import numpy as np
import math
from scipy import integrate
from constant import Constant

def update_x(x, U, p, t):
    x1, x2, x3 = x
    u1, u2, u3 = U
    p1, p2, p3 = p

    x11, x12 = x1
    x21, x22 = x2
    x31, x32 = x3

    p11, p12, p13 = p1
    p21, p22, p23 = p2
    p31, p32, p33 = p3

    p1_ = (p11 * x12 * np.sin(x21) * math.cos(t) +
           p12 * x11 * np.sin(x21 ** 2) +
           p13 * x12 * np.cos(x31))[0]
    p2_ = (p21 * x21 * math.sin(t) +
           p22 * x22 * np.cos(x12) +
           p23 * x22 * np.sin((x32)))[0]
    p3_ = (p32 * x32 * np.cos(x11) * math.sin(t) +
           p32 * x32 * (np.sin(x21) ** 2) +
           p33 * x12 * np.cos(x22))[0]

    # print(U, [p1_, p2_, p3_])

    x1_ = (np.array([-x12, x11 * x12])
           + np.array([[0], 2 - np.cos(x12) ** 2]) * (u1 + p1_))

    x1_ = np.clip(x1_, -1, 1)

    x2_ = (np.array([x22, -x21 - 0.5 * x22 + 0.5 * (x21 ** 2) * x22])
           + np.array([[0], 1 + np.cos(x21)]) * (u2 + p2_))
    x2_ = np.clip(x2_, -1, 1)

    x3_ = (np.array([-x31 + x32, -0.5 * (x31 + x32)])
           + np.array([[0], 0.5 * x32 * ((2 + np.cos(2 * x31) ** 2) ** 2) + (2 + np.cos(2 * x21) ** 2)])* (u3 + p3_))
    x3_ = np.clip(x3_, -1, 1)

    return x1_, x2_, x3_


def update_xd(xd):
    xd1, xd2, xd3 = xd
    xd1_ = np.matmul(np.array([[0, -1], [0.25, 0]]), xd1)
    xd2_ = np.matmul(np.array([[0, 1], [-1, 0]]), xd2)
    xd3_ = np.matmul(np.array([[-1, 1], [-2, 1]]), xd3)
    return [xd1_, xd2_, xd3_]

def Gx(X):
    X1, X2, X3 = X
    X11, X12, X13, X14 = X1
    X21, X22, X23, X24 = X2
    X31, X32, X33, X34 = X3

    Gx1 = 2 - np.cos(X12 + X14) ** 2
    Gx2 = 1 + np.cos(X21 + X23)
    Gx3 = 0.5 * (X32 + X34) * (2 + np.cos((2 * (X31 + X33)) ** 2)) + 2 + np.cos(2 + np.cos(2 * (X31 + X33)) ** 2) ** 2
    return Gx1[0], Gx2[0], Gx3[0]


def update_X(X1, X2, X3, u1, u2, u3):
    X11, X12, X13, X14 = X1
    X21, X22, X23, X24 = X2
    X31, X32, X33, X34 = X3

    X1_ = (np.array([-X11,
                    (X11 + X13) * (X12 + X14) + 0.25 * X13,
                    -X14,
                    0.25 * X13])
           + (np.array([[0], (2 - np.cos(X12 + X14) ** 2), [0], [0]]) * u1))
    X2_ = (np.array([X22,
                    -X21 - 0.5 * (X22 + X24) + 0.5 * ((X21 + X23) ** 2),
                    X24,
                    -X23])
           + (np.array([[0], (1 + np.cos(X21 + X23)), [0], [0]]) * u2))
    X3_ = (np.array([-X31 + X32,
                    -0.5 * (X31 + X32 + X33 + X34) + 2 * X33 - X34,
                    -X33 + X34,
                    -2 * X33 + X34])
           + (np.array([[0], (0.5 * (X32 + X34) * (2 + np.cos((2 * (X31 + X33)) ** 2)) +
                              2 + np.cos(2 + np.cos(2 * (X31 + X33)) ** 2) ** 2), [0], [0]]) * u3))
    return X1_, X2_, X3_

def compute_ei(xi, xid):
    return xi - xid

def compute_X(ei, xdi):
    return np.concatenate([ei, xdi], axis=0)

def compute_x_from_X(Xi):
    ei = Xi[:2]
    xdi = Xi[2:]
    return ei + xdi

def compute_e_xd_from_X(Xi):
    ei = Xi[:2]
    xdi = Xi[2:]
    return ei, xdi

def compute_VX(X, u, t):
    # print(u, u/alpha, np.arctanh(u/alpha).T)

    ratio = (np.matmul(np.matmul(X.T, Constant.Q), X)[0, 0]
             + 2 * Constant.alpha * Constant.R * (np.arctanh(u/Constant.alpha).T * u)
             # + 2 * Constant.alpha * Constant.R * (max(min(np.arctanh(u/Constant.alpha).T, 1000.0), -1000.0) * u)
             + Constant.alpha ** 2 * Constant.R * np.log(1 - (u[0] / Constant.alpha) ** 2))
    # print(np.dot(np.dot(X.T, Q), X), 2 * alpha * R * np.dot(np.arctanh(u/alpha).T, u), alpha ** 2 * R * np.log(1 - (u / alpha) ** 2))
    # print(X, u, t, ratio)
    def f(tau):
        ret = (math.exp(-Constant.lamb * (tau - t)) * ratio)
        return ret


    v = integrate.quad(f, 0, float('inf'))[0]
    return min(v, 20000)


def compute_deltaV(X, X_next, u, V):

    def f(tau):
        # ret = (max(min(np.arctanh(u/Constant.alpha).T, 1000.0), -1000.0)) * Constant.R
        ret = np.arctanh(u/Constant.alpha).T * Constant.R
        return ret

    pre = np.matmul(np.matmul(X.T, Constant.Q), X)[0,0] + 2 * Constant.alpha * integrate.quad(f, 0, u[0])[0] - Constant.lamb * V
    # print(pre, X_next)
    return pre/X_next
