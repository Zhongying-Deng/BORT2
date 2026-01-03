import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pdb
from collections import OrderedDict
from sklearn.manifold import TSNE

def plot_embedding(X, y, d, fig_mode='save', title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.
    :param X: embedding
    :param y: label
    :param d: domain
    :param fig_mode: display or save as image
    :param title: title on the figure
    :param imgName: the name of saving image
    :return:
    """
    if fig_mode is None:
        return

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 #color=plt.cm.bwr(d[i] / 1.),
                 #color=plt.cm.rainbow(d[i] / 1.),
                 color=plt.cm.rainbow(y[i] / y.max()),
                 fontdict={'weight': 'bold', 'size': 10})  # d[i]=1, red; d[i]=0, blue

    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    #else:
    #    plt.title('Visualization')

    if fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()

    if fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath('./fig')

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        print('Saving ' + imgName + ' ...')
        plt.savefig(imgName)
        plt.savefig(imgName.replace('.jpg', '.pdf'))
        plt.close()


def tsne(feat_source, feat_target, s_labels, t_labels, imgName, domain_x=None, plot_tar_only=True, save=True, title=None,
         n_components=2, perplexity=30, n_iter=3000):
    tsne = TSNE(perplexity=perplexity, n_components=n_components, init='pca', n_iter=n_iter)
    if domain_x is None:
        s_domain_tags = torch.zeros_like(s_labels)  # source data is 0
        t_domain_tags = torch.ones_like(t_labels)  # target data is 1
    else:
        s_domain_tags = torch.cat(domain_x, 0)
        s_domain_tags = s_domain_tags / (s_domain_tags.max() + 1.)
        t_domain_tags = torch.ones_like(t_labels)

    if save:
        fig_mode = 'save'
    else:
        fig_mode = 'display'

    if not plot_tar_only:
        dann_tsne = tsne.fit_transform(np.concatenate((feat_source.detach().numpy(),
                                                           feat_target.detach().numpy())))
        plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                np.concatenate((s_domain_tags, t_domain_tags)), fig_mode, title=title, imgName=imgName)
    else:
        dann_tsne = tsne.fit_transform(feat_target.detach().numpy())
        plot_embedding(dann_tsne, t_labels.detach().numpy(), t_domain_tags.detach().numpy(), fig_mode, title=title, imgName=imgName)
