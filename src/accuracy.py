import numpy as np


def compute_pseudo_accuracy(model, triplets):
    anchors, positives, negatives = triplets
    predictions = model.predict([anchors, positives, negatives])
    print(predictions[:3])
    return np.mean(predictions[:, 0, 0] < predictions[:, 1, 0])
