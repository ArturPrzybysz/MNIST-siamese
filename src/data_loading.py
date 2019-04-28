import numpy as np

from src.config import NUM_CLASSES, INPUT_SHAPE, SEMI_HARD_IMG_COUNT, MARGIN
from src.util import vstack_matrices


def semi_hard_triplets(x, y, embedding_model, triplets_count):
    # TODO: check, if embedding models weights are the same object as in siamese
    x, y = unison_shuffled_copies(x[:SEMI_HARD_IMG_COUNT], y[:SEMI_HARD_IMG_COUNT])
    embeddings = embedding_model.predict(x)

    anchors = None
    positives = None
    negatives = None

    for i in range(NUM_CLASSES):
        class_anchors, class_positives, class_negatives = _semi_hard_triplets_per_class(i, embeddings, x, y,
                                                                                        triplets_count // NUM_CLASSES)
        anchors = vstack_matrices(anchors, class_anchors)
        positives = vstack_matrices(positives, class_positives)
        negatives = vstack_matrices(negatives, class_negatives)

    assert triplets_count == len(anchors) == len(positives) == len(negatives)
    return anchors, positives, negatives


def _semi_hard_triplets_per_class(class_idx, embeddings, x, y, triplets_count):
    # TODO: make balanced triplets
    class_indices = y == class_idx
    positive_xs = x[class_indices]
    positive_embeddings = embeddings[class_indices]
    anchor_embeddings, anchor_xs = unison_shuffled_copies(positive_embeddings, positive_xs)
    negative_xs = x[np.logical_not(class_indices)]
    negative_embeddings = x[np.logical_not(class_indices)]

    anchors = None
    positives = None
    negatives = None

    for anch_emb, anch_x in zip(anchor_embeddings, anchor_xs):
        for pos_emb, pos_x in zip(positive_embeddings, positive_xs):
            for neg_emb, neg_x in zip(negative_embeddings, negative_xs):
                if anchors is not None and len(anchors) == triplets_count:
                    break
                pos_dist = np.linalg.norm(anch_emb - pos_emb)
                neg_dist = np.linalg.norm(anch_emb - neg_emb)

                if pos_dist < neg_dist < pos_dist + MARGIN:
                    anchors = vstack_matrices(anchors, anch_x)
                    positives = vstack_matrices(positives, pos_x)
                    negatives = vstack_matrices(negatives, neg_x)

    anchors = np.array(anchors)
    positives = np.array(positives)
    negatives = np.array(negatives)

    if anchors.shape[0] != triplets_count:
        deficit = triplets_count - len(anchors)
        anchors = vstack_matrices(anchors, positive_xs[:deficit])
        positives = vstack_matrices(positives, positive_xs[len(positive_xs) - deficit:])
        random_negatives = positive_xs[np.random.choice(len(negative_xs), deficit, replace=False)]
        negatives = vstack_matrices(negatives, random_negatives)
    assert triplets_count == len(negatives) == len(positives) == len(negatives)
    return anchors, positives, negatives


def _triplets_per_class(class_idx, x, y, triplets_count):
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
        class_anchors, class_positives, class_negatives = _triplets_per_class(i, x, y, triplets_count // NUM_CLASSES)
        anchors = vstack_matrices(anchors, class_anchors)
        positives = vstack_matrices(positives, class_positives)
        negatives = vstack_matrices(negatives, class_negatives)

    assert triplets_count == len(anchors) == len(positives) == len(negatives)
    return anchors, positives, negatives


def preprocess(x):
    x = x.astype('float32')
    x /= 255
    x = x.reshape((len(x), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
    return x


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
