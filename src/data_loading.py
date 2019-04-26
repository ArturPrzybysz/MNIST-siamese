import numpy as np

from src.config import NUM_CLASSES, INPUT_SHAPE
from src.util import vstack_matrices


def preprocess(x):
    x = x.astype('float32')
    x /= 255
    x = x.reshape((len(x), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
    return x


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def _triples_per_class(class_idx, x, y, triplets_count):
    class_indices = y == class_idx
    positive_xs = x[class_indices]

    anchors = positive_xs[np.random.choice(len(positive_xs), triplets_count, replace=False)]
    positives = positive_xs[np.random.choice(len(positive_xs), triplets_count, replace=False)]
    negatives = None

    negative_xs_per_class = triplets_count // (NUM_CLASSES - 1)
    for negative_class_idx in np.arange(NUM_CLASSES):
        if negative_class_idx != class_idx:
            negative_class_indices = y == negative_class_idx
            negative_xs = x[negative_class_indices]

            chosen_negatives = negative_xs[np.random.choice(len(negative_xs), negative_xs_per_class, replace=False)]
            negatives = vstack_matrices(negatives, chosen_negatives)

    assert triplets_count == len(negatives) == len(positives) == len(negatives)
    return anchors, positives, negatives


def random_triplets(x, y, triplets_count):
    x, y = unison_shuffled_copies(x, y)

    anchors = None
    positives = None
    negatives = None

    for i in range(NUM_CLASSES):
        class_anchors, class_positives, class_negatives = _triples_per_class(i, x, y, triplets_count // NUM_CLASSES)
        anchors = vstack_matrices(anchors, class_anchors)
        positives = vstack_matrices(positives, class_positives)
        negatives = vstack_matrices(negatives, class_negatives)

    assert triplets_count == len(anchors) == len(positives) == len(negatives)
    return anchors, positives, negatives
