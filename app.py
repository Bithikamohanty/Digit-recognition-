import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognition")
st.title("Handwritten Digit Recognition")

model = tf.keras.models.load_model("model.h5")

def saliency_map(model, img_array):
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, img_tensor)
    grads = tf.abs(grads)
    grads /= tf.reduce_max(grads) + 1e-8
    heatmap = grads[0, :, :, 0].numpy()
    return heatmap

canvas = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    img = canvas.image_data[:, :, 0]
    img = Image.fromarray(img).resize((28, 28))
    img = np.array(img)
    img = img / 255.0
    img_array = img.reshape(1, 28, 28, 1)

    predictions = model.predict(img_array)
    pred_digit = np.argmax(predictions)
    confidence = predictions[0]

    st.write("Prediction:", pred_digit)
    for i, p in enumerate(confidence):
        st.progress(float(p))
        st.write(i, float(p))

    heatmap = saliency_map(model, img_array)
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((280, 280))
    st.image(heatmap_img, caption="Saliency Map", width=280)
