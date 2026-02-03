import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def explain_prediction(model, img):
    img_tensor = tf.convert_to_tensor(img)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
        class_idx = tf.argmax(prediction[0])
        loss = prediction[:, class_idx]

    gradients = tape.gradient(loss, img_tensor)
    heatmap = tf.reduce_mean(tf.abs(gradients), axis=-1).numpy()

    return heatmap[0]
