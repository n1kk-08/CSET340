import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image, color=False):
    if color:
        channels = ('b', 'g', 'r')
        for i, col in enumerate(channels):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    plt.xlabel('Gray Level / Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram')
    plt.show()

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def clahe_equalization_color(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return equalized_image


def compute_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    return dft, dft_shift, magnitude_spectrum

def inverse_fft(dft):
    dft_shift_inv = np.fft.ifftshift(dft)
    img_reconstructed = np.fft.ifft2(dft_shift_inv)
    return np.abs(img_reconstructed)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Load images
gray_image = cv2.imread('GrayScale.png', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('Colour_Img.png')

gray_equalized = histogram_equalization(gray_image)

color_equalized_clahe = clahe_equalization_color(color_image)

# Display histogram
compute_histogram(gray_image)
compute_histogram(gray_equalized)

compute_histogram(color_image, color=True)
compute_histogram(color_equalized_clahe, color=True)

# FFT and IFFT
dft, dft_shift, magnitude_spectrum = compute_fft(gray_image)
reconstructed_image = inverse_fft(dft)
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(133), plt.imshow(reconstructed_image, cmap='gray'), plt.title('Reconstructed')
plt.show()

# Rotation and FFT comparison
rotated_image = rotate_image(gray_image, 45)
_, _, rotated_magnitude_spectrum = compute_fft(rotated_image)
plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Original Spectrum')
plt.subplot(122), plt.imshow(rotated_magnitude_spectrum, cmap='gray'), plt.title('Rotated Spectrum')
plt.show()