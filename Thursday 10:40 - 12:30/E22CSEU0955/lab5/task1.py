import cv2
import numpy as np
import matplotlib.pyplot as plt


def blockwise_dct(image, block_size=8):
    h, w = image.shape
    dct_image = np.zeros_like(image, dtype=np.float32)


    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            dct_block = cv2.dct(np.float32(block))
            dct_image[i:i + block_size, j:j + block_size] = dct_block

    return dct_image

def blockwise_idct(dct_image, block_size=8):
    h, w = dct_image.shape
    image_reconstructed = np.zeros_like(dct_image, dtype=np.float32)


    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dct_block = dct_image[i:i + block_size, j:j + block_size]
            block = cv2.idct(dct_block)
            image_reconstructed[i:i + block_size, j:j + block_size] = block

    return image_reconstructed


def blockwise_quantize_dct(dct_image, quant_matrix, block_size=8):
    h, w = dct_image.shape
    quantized_dct = np.zeros_like(dct_image, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_image[i:i + block_size, j:j + block_size]
            quantized_block = np.round(block / quant_matrix)  # Quantization
            quantized_dct[i:i + block_size, j:j + block_size] = quantized_block

    return quantized_dct

def blockwise_dequantize_dct(quantized_dct, quant_matrix, block_size=8):
    h, w = quantized_dct.shape
    dequantized_dct = np.zeros_like(quantized_dct, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_dct[i:i + block_size, j:j + block_size]
            dequantized_block = block * quant_matrix  # Dequantization
            dequantized_dct[i:i + block_size, j:j + block_size] = dequantized_block

    return dequantized_dct


image_path = "a.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


h, w = image.shape
new_h = (h + 7) // 8 * 8
new_w = (w + 7) // 8 * 8
padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=0)


quant_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


dct_image = blockwise_dct(padded_image)


quantized_dct = blockwise_quantize_dct(dct_image, quant_matrix)


dequantized_dct = blockwise_dequantize_dct(quantized_dct, quant_matrix)


image_reconstructed = blockwise_idct(dequantized_dct)


image_reconstructed = image_reconstructed[:h, :w]

image_reconstructed = np.clip(image_reconstructed, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_reconstructed, cmap='gray')
plt.title('Compressed Image')
plt.axis('off')

import os


original_image_path = "original_image.jpg"
cv2.imwrite(original_image_path, image)

compressed_image_path = "compressed_image.jpg"
cv2.imwrite(compressed_image_path, image_reconstructed)


original_size = os.path.getsize(original_image_path)
compressed_size = os.path.getsize(compressed_image_path)


print(f"Original Image Size: {original_size / 1024:.2f} KB")
print(f"Compressed Image Size: {compressed_size / 1024:.2f} KB")
print(f"Compression Ratio: {original_size / compressed_size:.2f}")


plt.show()