import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorflow import keras
from tensorflow.keras import layers
from ultralytics import YOLO
from torchvision import transforms, models
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from PIL import Image

def edge_based_segmentation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge-based Segmentation')
    plt.show()
    return edges

def region_based_segmentation(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, segmented = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    plt.imshow(segmented, cmap='gray')
    plt.title('Region-based Segmentation')
    plt.show()
    return segmented

def hough_transform(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 68, minLineLength=15, maxLineGap=50)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(output)
    plt.title('Hough Transform Line Detection')
    plt.show()

def gabor_lbp_texture(image_path):
    # 1. Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Apply Gabor filter
    ksize = 31
    sigma = 4.0
    theta = np.pi / 4  # Orientation = 45 degrees
    lamda = 10.0
    gamma = 0.5
    phi = 0
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    gabor_filtered = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

    # 3. Apply Local Binary Pattern (LBP)
    radius = 1  # LBP radius
    n_points = 8 * radius  # Number of points around
    lbp = local_binary_pattern(gabor_filtered, n_points, radius, method='uniform')

    # 4. Plot Original, Gabor Filtered, LBP Result
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1,3,2)
    plt.imshow(gabor_filtered, cmap='gray')
    plt.title('Gabor Filtered')
    
    plt.subplot(1,3,3)
    plt.imshow(lbp, cmap='gray')
    plt.title('LBP after Gabor')
    
    plt.show()


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.0
    x_test = np.expand_dims(x_test, axis=-1) / 255.0
    return (x_train, y_train), (x_test, y_test)


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))

    # Convert image to RGB if it has 4 channels (RGBA)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def object_detection_yolo(image_path):
    model = YOLO('yolov8n.pt')
    results = model(image_path, save=True, show=True)
    return results

def object_detection_rcnn(image_path):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image_np = np.array(image)  # Convert to numpy array

    image_tensor = preprocess_image(image_np)  # Preprocess image before passing to model
    
    with torch.no_grad():
        predictions = model(image_tensor)

    # Draw bounding boxes
    for i in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][i].numpy()
        score = predictions[0]['scores'][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(image_np)
    plt.title('R-CNN Object Detection')
    plt.show()


def classify_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test, y_test))
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("Final Classification Report for Fashion-MNIST (RCNN-based):")
    print(classification_report(y_test, y_pred))
    return model

def classify_cifar100():
    (x_train, y_train), (x_test, y_test) = load_cifar100()
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(100, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test, y_test))
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("Final Classification Report for CIFAR-100 (RCNN-based):")
    print(classification_report(y_test, y_pred))
    return model

def process_image(image_path_hough, image_path_yolo):
    print("Performing Edge-based Segmentation...")
    edge_based_segmentation(image_path_hough)
    print("Performing Region-based Segmentation...")
    region_based_segmentation(image_path_hough)
    print("Applying Hough Transform...")
    hough_transform(image_path_hough)
    print("Applying texture enhancement using Gabor and LBP...")
    gabor_lbp_texture(image_path_gabor_lbp)
    print("Running YOLO on Image...")
    object_detection_yolo(image_path_yolo)
    print("Running R-CNN on Image...")
    object_detection_rcnn(image_path_yolo)
    # print("Training Fashion-MNIST classifier...")
    # classify_fashion_mnist()
    # print("Training CIFAR-100 classifier...")
    # classify_cifar100()

image_path_hough = 'Hough_img.png'  
image_path_gabor_lbp="image_gabor.jpeg"
image_path_yolo = 'YOLO_img.png'  
process_image(image_path_hough, image_path_yolo)
