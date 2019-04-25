import numpy as np
from PIL import Image


def vstack_matrices(a, b):
    if a is None:
        return b
    else:
        return np.vstack((a, b))


def save_as_img(matrix, filename):
    img = Image.fromarray(matrix * 255)
    if img.mode != "RGB":
        img = img.convert("L")
    img.save(filename)
