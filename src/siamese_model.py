from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from tensorflow.python.keras import backend as K

from src.config import MARGIN


def siamese_model(input_shape, embedding_size):
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    embedder = _embedding_model(input_shape, embedding_size)

    anchor_embedding = embedder(anchor_input)
    positive_embedding = embedder(positive_input)
    negative_embedding = embedder(negative_input)

    L2_dist = Lambda(_euclidean_distance, name='L2_dist')

    positive_dist = L2_dist([anchor_embedding, positive_embedding])
    negative_dist = L2_dist([anchor_embedding, negative_embedding])

    stacked_dists = Lambda(lambda v: K.stack(v, axis=1), name='stacked_dists')([positive_dist, negative_dist])

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[stacked_dists])
    model.compile(optimizer=Adam(), loss=_triplet_loss)
    return model


def embedding_model(input_shape, embedding_size):
    model = _embedding_model(input_shape, embedding_size)
    model.compile(optimizer=None)
    return model


def _embedding_model(input_shape, embedding_size):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (4, 4), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (2, 2), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Dense(embedding_size)(x)
    output = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1), name='normalized_embedding')(x)

    model = Model(inputs=[inputs], outputs=[output])
    return model


def _euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def _triplet_loss(_, y_pred):
    margin = K.constant(MARGIN)
    positive_dist = y_pred[:, 0]
    negative_dist = y_pred[:, 1]

    basic_loss = K.square(positive_dist) - K.square(negative_dist) + margin

    return K.mean(K.maximum(K.constant(0), basic_loss))
