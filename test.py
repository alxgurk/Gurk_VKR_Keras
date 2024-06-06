import sys
import keras_cv
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from googletrans import Translator
from tensorflow import keras
import tensorflow as tf
import math
from PIL import Image

images = []
model = keras_cv.models.StableDiffusion()
prompt = "Автомат из пиццы"
encoding = tf.squeeze(model.encode_text(prompt))
walk_steps = 1
batch_size = 1
batches = walk_steps // batch_size
seed = 12345

noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

walk_noise_x = tf.random.normal(noise.shape, dtype=tf.float64)
walk_noise_y = tf.random.normal(noise.shape, dtype=tf.float64)

walk_scale_x = tf.cos(tf.linspace(0, 2, walk_steps) * math.pi)
walk_scale_y = tf.sin(tf.linspace(0, 2, walk_steps) * math.pi)
noise_x = tf.tensordot(walk_scale_x, walk_noise_x, axes=0)
noise_y = tf.tensordot(walk_scale_y, walk_noise_y, axes=0)
noise = tf.add(noise_x, noise_y)
batched_noise = tf.split(noise, batches)

for batch in range(batches):
    images += [
        Image.fromarray(img)
        for img in model.generate_image(
            encoding,
            batch_size=batch_size,
            num_steps=2,
            diffusion_noise=batched_noise[batch],
        )
    ]

images[0].save(
    fr"C:\Users\hp\PycharmProjects\qt\results\result.gif",
    save_all=True,
    duration=10,
    loop=0,
)
