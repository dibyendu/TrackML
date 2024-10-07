import numpy as np
from skimage.measure import block_reduce

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.legend_handler import HandlerTuple

def getXY(A):
    xind = range(A.shape[0])
    yind = range(A.shape[1])
    xx = []
    yy = []
    for p in zip(xind, A):
        for i in yind:
            if p[1][i] > 0:
                xx.append(p[0])
                yy.append(i)
    return xx, yy

def plotXY(X, Y):
    plt.scatter(X, Y)
    plt.show()

def plotLayers(hL, show=False):
    for p in hL:
        plt.figure()
        z, phi = getXY(p)
        plt.scatter(phi, z)
        plt.xlim(0, p.shape[1])
        plt.ylim(0, p.shape[0])

    if show:
        for i in plt.get_fignums():
            plt.show(i)

def plotLayersSinglePlot(hL, show=False, title='Layer', save_loc=None):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    bar = ['lightblue', 'skyblue','deepskyblue', 'royalblue', 'blue', 'navy']
    fig = plt.figure()
    plt.title(title)
    legends, labels = [], []
    for i in range(len(hL)):
        p = hL[i]
        x, y = getXY(p)
        plt.scatter(x, y, color=colors[bar[i]], label='HL' + str(i))
        plt.xlim(0, p.shape[0])
        plt.ylim(0, p.shape[1])
        plt.xticks([])
        plt.yticks([])
        legends.append((plt.scatter([0], [0], color=colors[bar[i]], marker='o', s=40), ))
        labels.append('Layer ' + str(i+1))
    plt.legend(legends, labels, handler_map={tuple: HandlerTuple(ndivide=None)})

    if show:
        for i in plt.get_fignums():
            plt.legend()
            plt.show(block=True)
    
    if save_loc:
        plt.savefig(f'{save_loc}.png')
        plt.savefig(f'{save_loc}.svg', format='svg')
        plt.close(fig)

def PlotImage(hL, gthL, title1="Input Image", title2="Ground Truth Image", show=False):
    plotLayersSinglePlot(hL, show=show)
    plt.title(title1)
    plt.legend()
    plotLayersSinglePlot(gthL, show=show)
    plt.title(title2)
    plt.legend()

def PlotSingleImage(data, width=1024, title='Labelled Input Image', save_loc=None):
    hL, gthL = [], []
    for l in data['hL']:
        zeros = np.zeros((1024, 1024))
        zeros[tuple(np.array(list(l)).T.tolist())] = 1
        hL.append(zeros)
    for l in data['gthL']:
        zeros = np.zeros((1024, 1024))
        zeros[tuple(np.array(list(l)).T.tolist())] = 1
        gthL.append(zeros)
    hL, gthL = np.array(hL), np.array(gthL)

    if width < 1024:
        ratio = 1024 / width
        hL = np.heaviside(
            np.array(map(
                lambda x: block_reduce(x, block_size=(ratio,ratio), func=np.max),
                hL
            )), 0)
        gthL = np.heaviside(
            np.array(map(
                lambda x: block_reduce(x, block_size=(ratio,ratio), func=np.max),
                gthL
            )), 0)

    def getXY(A, layer_id):
        xind = range(A.shape[0])
        yind = range(A.shape[1])
        true_xx, true_yy, false_xx, false_yy = [], [], [], []
        for p in zip(xind, A):
            for i in yind:
                if p[1][i] > 0:
                    if p[1][i] == gthL[layer_id][p[0]][i]:
                        true_xx.append(p[0])
                        true_yy.append(i)
                    else:
                        false_xx.append(p[0])
                        false_yy.append(i)
        return true_xx, true_yy, false_xx, false_yy

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    true_color = ['palegreen', 'darkseagreen','mediumspringgreen', 'mediumseagreen', 'green', 'darkgreen']
    false_color = ['mistyrose', 'salmon','tomato', 'red', 'firebrick', 'darkred']
    legends, labels = [], []
    fig = plt.figure()
    for i in range(len(hL)):
        p = hL[i]
        tx, ty, fx, fy = getXY(p, i)
        plt.scatter(tx, ty, color=colors[true_color[i]])
        plt.scatter(fx, fy, color=colors[false_color[i]])
        plt.xlim(0, p.shape[0])
        plt.ylim(0, p.shape[1])
        plt.xticks([])
        plt.yticks([])
        legends.append((plt.scatter([0], [0], color=colors[true_color[i]], marker='o', s=40), plt.scatter([1], [0], color=colors[false_color[i]], marker='o', s=40)))
        labels.append('Layer ' + str(i+1))
    plt.title(title)
    plt.legend(legends, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    if save_loc:
        plt.savefig(f'{save_loc}.png')
        plt.savefig(f'{save_loc}.svg', format='svg')
        plt.close(fig)
    
def PlotModelPrediction(hL, noise_pos, missed, false_prediction, width=1024, title='Noise Prediction', save_loc=None):
    nhL = []
    for l in hL:
        zeros = np.zeros((1024, 1024))
        zeros[tuple(np.array(list(l)).T.tolist())] = 1
        nhL.append(zeros)
    nhL = np.array(nhL)

    if width < 1024:
        ratio = 1024 / width
        nhL = np.heaviside(
            np.array(map(
                lambda x: block_reduce(x, block_size=(ratio,ratio), func=np.max),
                nhL
            )), 0)

    def getXY(A, layer_id):
        xind = range(A.shape[0])
        yind = range(A.shape[1])
        true_xx, true_yy, false_xx, false_yy = [], [], [], []
        for p in zip(xind, A):
            for i in yind:
                if p[1][i] > 0:
                    if (p[0], i) not in noise_pos:
                        true_xx.append(p[0])
                        true_yy.append(i)
                    else:
                        false_xx.append(p[0])
                        false_yy.append(i)
        return true_xx, true_yy, false_xx, false_yy
        
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    point_color = ['lightblue', 'skyblue','deepskyblue', 'royalblue', 'blue', 'navy']
    noise_color = ['mistyrose', 'salmon','tomato', 'red', 'firebrick', 'darkred']
    legends, labels = [], []
    fig = plt.figure()
    if len(missed):
        plt.scatter([p[0] for p in missed], [p[1] for p in missed], s=100, facecolors='none', edgecolors='#000000')
    if len(false_prediction):
        plt.scatter([p[0] for p in false_prediction], [p[1] for p in false_prediction], s=100, facecolors='none', edgecolors='#000000')
    for i in range(len(nhL)):
        p = nhL[i]
        tx, ty, fx, fy = getXY(p, i)
        plt.scatter(tx, ty, color=colors[point_color[i]])
        plt.scatter(fx, fy, color=colors[noise_color[i]])
        plt.xlim(0, p.shape[0])
        plt.ylim(0, p.shape[1])
        plt.xticks([])
        plt.yticks([])
        legends.append((plt.scatter([0], [0], color=colors[point_color[i]], marker='o', s=40), plt.scatter([1], [0], color=colors[noise_color[i]], marker='o', s=40)))
        labels.append('Layer ' + str(i+1))
    plt.title(title)
    plt.legend(legends, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    if save_loc:
        plt.savefig(f'{save_loc}.png')
        plt.savefig(f'{save_loc}.svg', format='svg')
        plt.close(fig)