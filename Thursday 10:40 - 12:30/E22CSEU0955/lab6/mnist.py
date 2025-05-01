import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from ultralytics import YOLO
from PIL import Image

FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def load_fashion_mnist():
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.0
    return x_train, y_train

def preprocess_for_yolo(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (640, 640))
    return image

def detect_with_yolo(image_path):
    model = YOLO('yolov8n.pt')
    model.model.names = FASHION_CLASSES

    results = model(image_path)

    for r in results:
        im_array = r.plot()
        plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("YOLO Detection on Fashion MNIST")
        plt.show()

    return results

def main_yolo():
    fashion_images, _ = load_fashion_mnist()

    fashion_idx = random.randint(0, len(fashion_images) - 1)
    fashion_img = (fashion_images[fashion_idx] * 255).astype(np.uint8).squeeze()

    fashion_path = "fashion_sample.png"
    Image.fromarray(fashion_img).convert("RGB").save(fashion_path)

    plt.figure(figsize=(5, 5))
    plt.imshow(fashion_img, cmap='gray')
    plt.title("Fashion MNIST Sample")
    plt.axis('off')
    plt.show()

    print("\nRunning YOLO on Fashion MNIST image...")

    fashion_img_cv = cv2.imread(fashion_path, cv2.IMREAD_GRAYSCALE)
    fashion_img_processed = preprocess_for_yolo(fashion_img_cv)
    cv2.imwrite(fashion_path, fashion_img_processed)

    _ = detect_with_yolo(fashion_path)

if __name__ == "__main__":
    main_yolo()