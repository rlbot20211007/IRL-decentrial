import numpy as np
import matplotlib.pyplot as plt

def draw_value(Y, y_label):
    Y = np.array(Y)[1:]
    print(y_label, Y.shape)

    if len(Y.shape) >= 2:
        Y = Y.reshape(Y.shape[:2])

    for ind, y in enumerate(Y.T):
        # print(ind, y)
        plt.plot(y, label='system ' + str(ind))
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel(y_label)
    plt.savefig('results/' + y_label + '.png')
    # plt.show()
    plt.clf()


def draw_weight(Y, y_label):
    Y = np.array(Y)
    # print(y_label, Y.shape)

    if len(Y.shape) >= 2:
        Y = Y.reshape(Y.shape[:2])

    for ind, y in enumerate(Y.T):
        plt.plot(y)
        # print(y_label, ind, y)
    # plt.legend()
    plt.xlabel('iteration')
    plt.ylabel(y_label)
    plt.savefig('results/' + y_label + '.png')
    # plt.show()
    plt.clf()


def draw_x_xd(x_xd):
    for i in range(3):
        x11 = []
        x12 = []
        xd11 = []
        xd12 = []
        for data in x_xd:
            x, xd = data
            x11.append(x[i][0])
            x12.append(x[i][1])
            xd11.append(xd[i][0])
            xd12.append(xd[i][1])
        e11 = np.array(x11) - np.array(xd11)
        e12 = np.array(x12) - np.array(xd12)

        plt.subplot(211)
        plt.plot(x11[:5], label='x11')
        plt.plot(xd11, '--', label='xd11')

        plt.subplot(212)
        plt.plot(x12[:5], label='x12')
        plt.plot(xd12, '--', label='xd12')

        plt.legend()
        plt.xlabel('iteration')
        y_label = 'the tracking state trajectory of system '+str(i)
        plt.ylabel(y_label)
        plt.savefig('results/' + y_label + '.png')
        # plt.show()
        plt.clf()


        plt.subplot(211)
        plt.plot(e11[:5], label='e11')

        plt.subplot(212)
        plt.plot(e12[:5], label='e12')

        plt.legend()
        plt.xlabel('iteration')
        y_label = 'the tracking error of system  '+str(i)
        plt.ylabel(y_label)
        plt.savefig('results/' + y_label + '.png')
        # plt.show()
        plt.clf()



