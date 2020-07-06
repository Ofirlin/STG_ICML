import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_curve(L, title='Training Curve'):
    matplotlib.use('TkAgg')
    x = range(1, len(L) + 1)
    plt.plot(x, L)
    # no ticks
    plt.xticks([])
    plt.title(title)
    plt.show()

def plot_curve(L, L_valid, title='Training Curve'):
    matplotlib.use('TkAgg')
    x = range(1, len(L) + 1)
    plt.plot(x, L, label='train')
    plt.plot(x, L_valid, label='valid')
    # no ticks
    plt.xticks([])
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()

def save_plot(L, L_valid, output_dir, plot_name, title='Training Curve'):
    x = range(1, len(L) + 1)
    plt.plot(x, L, label='train')
    plt.plot(x, L_valid, label='valid')
    # no ticks
    plt.xticks([])
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(output_dir + '/' + plot_name + ".png")
    plt.gcf().clear()

def save_heatmap(nparray, output_dir, plot_name):
    plt.imshow(nparray)
    plt.colorbar()
    plt.savefig(output_dir+'/'+plot_name+'.png')
    plt.gcf().clear()

def plt_surLines(T, surRates):
    plt.plot(T, np.transpose(surRates))
    plt.show()

