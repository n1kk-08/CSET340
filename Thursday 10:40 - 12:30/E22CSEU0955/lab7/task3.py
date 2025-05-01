import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16
import time

# Load CIFAR-100
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# VGG16 Model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16 = models.Sequential([
    vgg_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(100, activation='softmax')
])

vgg16.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Light AlexNet Model
def build_light_alexnet(input_shape=(32, 32, 3), num_classes=100):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(100, activation='softmax')
    ])
    return model

alexnet_light = build_light_alexnet()

alexnet_light.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train both models
vgg16.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
alexnet_light.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate both models
vgg16_eval = vgg16.evaluate(x_test, y_test, verbose=0)
alexnet_eval = alexnet_light.evaluate(x_test, y_test, verbose=0)

print(f"VGG16 - Loss: {vgg16_eval[0]:.4f}, Accuracy: {vgg16_eval[1]:.4f}")
print(f"Light AlexNet - Loss: {alexnet_eval[0]:.4f}, Accuracy: {alexnet_eval[1]:.4f}")

# Inference Timing
start_time = time.time()
_ = vgg16.predict(x_test[:10])
vgg16_time = time.time() - start_time

start_time = time.time()
_ = alexnet_light.predict(x_test[:10])
alexnet_time = time.time() - start_time

print(f"VGG16 Inference Time: {vgg16_time:.4f} seconds")
print(f"Light AlexNet Inference Time: {alexnet_time:.4f} seconds")