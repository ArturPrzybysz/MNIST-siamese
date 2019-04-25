from tensorflow.python.keras.datasets import mnist

from src.accuracy import compute_pseudo_accuracy
from src.config import TRAIN_TRIPLES, INPUT_SHAPE, EMBEDDING_SIZE, EPOCHS, BATCH_SIZE
from src.data_loading import random_triplets, preprocess
from src.siamese_model import siamese_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = preprocess(x_train)
x_test = preprocess(x_test)

anchors, positives, negatives = random_triplets(x_train, y_train, TRAIN_TRIPLES)
valid_triplets = random_triplets(x_test, y_test, TRAIN_TRIPLES)

model = siamese_model(INPUT_SHAPE, EMBEDDING_SIZE)
print(compute_pseudo_accuracy(model, valid_triplets))

for i in range(EPOCHS):
    model.fit([anchors, positives, negatives], batch_size=BATCH_SIZE, epochs=5)
    anchors, positives, negatives = random_triplets(x_train, y_train, TRAIN_TRIPLES)
    print(compute_pseudo_accuracy(model, valid_triplets))
