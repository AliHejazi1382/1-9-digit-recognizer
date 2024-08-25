import numpy as np
import tensorflow as tf
from recognizer import *


X, y = load_data()
m, n = X.shape

model = get_model(n)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

model.fit(X, y, epochs=100)
showResults(X, y, model)