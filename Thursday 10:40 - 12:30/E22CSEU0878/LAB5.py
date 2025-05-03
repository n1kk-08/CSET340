import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, cifar10

# Task 1: Image Compression


def compress_image(input_path, output_path, quality, format):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not load image!")
        return
    
    img = np.float32(img)  # Keep pixel range in [0, 255] but float for DCT

    if format == 'JPEG':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb
        Y, Cr, Cb = cv2.split(img)  # Separate channels
    else:  
        # PNG (Apply DCT separately on R, G, and B channels)
        Y, Cr, Cb = cv2.split(img)

    # Define JPEG Quantization Matrix (scaled based on quality)
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]) * (100 - quality) / 50  # Scale based on quality

    def dct_compress_channel(channel):
        """ Applies DCT compression block-wise on a single channel """
        h, w = channel.shape
        h_pad = 8 - (h % 8) if h % 8 != 0 else 0
        w_pad = 8 - (w % 8) if w % 8 != 0 else 0

        channel_padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='constant')
        h_padded, w_padded = channel_padded.shape
        compressed_channel = np.zeros_like(channel_padded)

        for y in range(0, h_padded, 8):
            for x in range(0, w_padded, 8):
                block = channel_padded[y:y+8, x:x+8] - 128  # Shift for DCT
                dct_block = cv2.dct(block)
                quantized_block = np.round(dct_block / Q) * Q
                idct_block = cv2.idct(quantized_block) + 128  # Shift back
                compressed_channel[y:y+8, x:x+8] = idct_block

        return compressed_channel[:h, :w]  # Remove padding

    # Apply DCT compression
    Y_compressed = dct_compress_channel(Y)
    if format == 'JPEG':
        # JPEG: Keep Cr and Cb unchanged
        img_compressed = cv2.merge([Y_compressed, Cr, Cb])
        img_compressed = cv2.cvtColor(img_compressed, cv2.COLOR_YCrCb2BGR)  # Convert back to BGR
    else:
        # PNG: Apply DCT to all channels
        Cr_compressed = dct_compress_channel(Cr)
        Cb_compressed = dct_compress_channel(Cb)
        img_compressed = cv2.merge([Y_compressed, Cr_compressed, Cb_compressed])

    # Normalize back to 0-255
    img_compressed = np.clip(img_compressed, 0, 255).astype(np.uint8)

    # Save compressed image
    if format == 'JPEG':
        cv2.imwrite(output_path, img_compressed, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif format == 'PNG':
        cv2.imwrite(output_path, img_compressed, [int(cv2.IMWRITE_PNG_COMPRESSION), 9 - quality // 10])

    print(f"DCT compressed {format} image saved at {output_path}")


# Task 2: CNN for MNIST and CIFAR-10

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")
    return accuracy, precision, recall, f1, cm


input_image="Colour_Img.png"
op_path_jpeg="compressed_img.jpg"
op_path_png="compressed_img.png"
jpeg_quality=50
png_compression=50

compress_image(input_image, op_path_jpeg, jpeg_quality, "JPEG")

compress_image(input_image, op_path_png, png_compression, "PNG")

# # MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model_mnist = build_cnn((28,28,1), 10)
model_mnist.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=128)
evaluate_model(model_mnist, X_test, y_test)

# CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model_cifar = build_cnn((32,32,3), 10)
model_cifar.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=128)
evaluate_model(model_cifar, X_test, y_test)
