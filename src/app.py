from tensorflow.python.keras.datasets import mnist
import numpy as np
from src.accuracy import compute_pseudo_accuracy
from src.config import TRAIN_TRIPLES, INPUT_SHAPE, EMBEDDING_SIZE, EPOCHS, BATCH_SIZE, EPOCHS_PER_BATCH
from src.data_loading import random_triplets, preprocess, semi_hard_triplets
from src.plot import plot_TSNE
from src.siamese_model import siamese_model, embedding_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = preprocess(x_train)
x_test = preprocess(x_test)

anchors, positives, negatives = random_triplets(x_train, y_train, TRAIN_TRIPLES)
valid_triplets = random_triplets(x_test, y_test, TRAIN_TRIPLES)

model = siamese_model(INPUT_SHAPE, EMBEDDING_SIZE)
print(compute_pseudo_accuracy(model, valid_triplets))
dummy_y = np.zeros((len(anchors), 3, 1))

for i in range(EPOCHS):
    model.fit([anchors, positives, negatives], dummy_y, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_BATCH)
    embed_model = embedding_model(INPUT_SHAPE, EMBEDDING_SIZE, model)
    anchors, positives, negatives = random_triplets(x_train, y_train, TRAIN_TRIPLES)
    print(compute_pseudo_accuracy(model, valid_triplets))

emb_model = embedding_model(INPUT_SHAPE, EMBEDDING_SIZE, model)

prediction = emb_model.predict(x_test[:1000])
plot_TSNE(prediction, y_test[:1000])
prediction = emb_model.predict(x_train[:1000])
plot_TSNE(prediction, y_train[:1000])
