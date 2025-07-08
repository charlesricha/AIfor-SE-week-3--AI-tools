import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("🖌️ Handwritten Digit Recognizer")
st.write("Upload an image of a single digit (0–9), and I’ll try to guess it!")

uploaded_file = st.file_uploader("📤 Upload a digit image", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        st.success(f"🎯 I think this is a **{predicted_digit}**!")
