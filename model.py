import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.python.layers.pooling import MaxPooling2D

tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test =x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from tensorflow.keras import Sequential, layers
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

fashion_labels = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {fashion_labels[predicted_labels[i]]} \nActual: {fashion_labels[y_test[i]]}")
    plt.axis('off')
plt.show()

import os
model.export('saved_models')
export_path = os.path.join('saved_models', 'fashion_classifier', '1')
os.makedirs(export_path, exist_ok=True)
model.export(export_path)
print(f'Model successfully exported to: {export_path}')

import gradio as gr

def predict_clothing(image):
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    preds = model.predict(image)[0]  # Get prediction vector
    top_idx = np.argmax(preds)
    return {fashion_labels[i]: float(preds[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict_clothing,
    inputs=gr.Image(image_mode="L"),
    outputs=gr.Label(num_top_classes=3),
    title="👕 Fashion Classifier",
    description="Draw or upload a 28x28 grayscale image of a clothing item. The model will predict the category."
)

demo.launch()



