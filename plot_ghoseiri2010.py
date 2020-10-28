import operator

import matplotlib.pyplot as plt


def plot(coorX, coorY, path):

    plt.plot(coorX, coorY, 'co')

    count = -1

    for _ in range(1, len(path)):
        i = path[_ - 1]
        j = path[_]
        # noinspection PyUnresolvedReferences
        #plt.arrow(coorX[i], coorY[i], coorX[j] - coorX[i], coorY[j] - coorY[i], color='r', length_includes_head=True)

        count +=1

        offsetX=0.0000075; offsetY=0.000025
        texto =str(count)
        plt.text(coorX[i-1] + offsetX, coorY[i-1], texto, rotation=0, size=8)

        plt.plot([coorX[i-1],coorX[j-1]],[coorY[i-1],coorY[j-1]],'r--')

    # noinspection PyTypeChecker
    #plt.xlim(0, max(coorX) * 1.1)
    # noinspection PyTypeChecker
    #cplt.ylim(0, max(coorY) * 1.1)
    plt.show()
