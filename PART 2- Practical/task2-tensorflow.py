# task2-tensorflow.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 📦 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 🧼 2. Normalize and reshape
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., np.newaxis]  # shape: (num, 28, 28, 1)
x_test = x_test[..., np.newaxis]

# 🧠 3. Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 🧪 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 🏋️ 5. Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# ✅ 6. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")

# 🔥 Check if goal is achieved
if test_acc >= 0.95:
    print("✅ Goal achieved! Accuracy > 95%")
else:
    print("⚠️ Keep training — not there yet!")

# 💾 7. Save the model
model.save("mnist_cnn.h5")
print("📁 Model saved as mnist_cnn.h5")

# 🖼️ 8. Visualize predictions on 5 sample images
sample_indices = np.random.choice(len(x_test), 5, replace=False)

plt.figure(figsize=(10, 2))
for i, idx in enumerate(sample_indices):
    img = x_test[idx]
    true_label = y_test[idx]
    pred_label = np.argmax(model.predict(img[np.newaxis, ...]))

    plt.subplot(1, 5, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nDone You are set to go! 😎")
