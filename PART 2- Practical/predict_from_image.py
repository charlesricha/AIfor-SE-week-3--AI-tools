# predict_from_image.py

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# 🧠 Load the trained model
model = tf.keras.models.load_model("mnist_cnn.h5")
print("✅ Model loaded!")

# add your image file here
img_path = "number.jpg"

# 🧼 Preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Grayscale
    img = ImageOps.invert(img)               # Invert to match MNIST (white digit, black bg)
    img = img.resize((28, 28))               # Resize
    img = np.array(img).astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)       # Add channel dim
    img = np.expand_dims(img, axis=0)        # Add batch dim
    return img

# 🔍 Predict
image = preprocess_image(img_path)
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)

# 🔢 Output
print(f"\n🧮 Predicted digit: {predicted_digit}")

# 🖼️ Show image
original_img = Image.open(img_path).convert('L')
plt.imshow(original_img, cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()
