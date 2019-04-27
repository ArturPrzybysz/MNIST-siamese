import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.config import NUM_CLASSES


def plot_TSNE(prediction, y):
    tsne = TSNE()
    pred_2d = tsne.fit_transform(prediction)

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i, c, label in zip(np.arange(NUM_CLASSES), colors, names):
        plt.scatter(pred_2d[y == i, 0], pred_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
