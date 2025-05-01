import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras
import torch
from torchvision import transforms, models
from PIL import Image

def load_cifar100():
    (x_train, y_train), _ = keras.datasets.cifar100.load_data()
    x_train = x_train / 255.0
    return x_train, y_train

def preprocess_for_rcnn(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def detect_with_faster_rcnn(image_path):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_tensor = preprocess_for_rcnn(image_np)

    with torch.no_grad():
        predictions = model(image_tensor)

    for i in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][i].numpy()
        score = predictions[0]['scores'][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{score:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(image_np)
    plt.axis('off')
    plt.title("Faster R-CNN Detection on CIFAR-100")
    plt.show()

    return predictions

def main_rcnn():
    cifar_images, _ = load_cifar100()
    cifar_idx = random.randint(0, len(cifar_images) - 1)
    cifar_img = (cifar_images[cifar_idx] * 255).astype(np.uint8)
    cifar_path = "cifar_sample.png"
    Image.fromarray(cifar_img).save(cifar_path)

    plt.figure(figsize=(5, 5))
    plt.imshow(cifar_img)
    plt.title("CIFAR-100 Sample")
    plt.axis('off')
    plt.show()

    print("\nRunning Faster R-CNN on CIFAR-100...")
    _ = detect_with_faster_rcnn(cifar_path)

if __name__ == "__main__":
    main_rcnn()