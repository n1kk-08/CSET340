import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    return dft, dft_shift, magnitude_spectrum

def compute_ifft(dft_shift):
    f_ishift = np.fft.ifftshift(dft_shift)
    img_reconstructed = np.fft.ifft2(f_ishift)
    return np.abs(img_reconstructed)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def process_fft(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, dft_shift, magnitude_spectrum = compute_fft(image)
    reconstructed_image = compute_ifft(dft_shift)

    rotated_image = rotate_image(image, 45)
    _, _, rotated_magnitude_spectrum = compute_fft(rotated_image)

    _, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(magnitude_spectrum, cmap='gray')
    axes[0, 1].set_title("Magnitude Spectrum")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(reconstructed_image, cmap='gray')
    axes[0, 2].set_title("Reconstructed Image")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(rotated_image, cmap='gray')
    axes[1, 0].set_title("Rotated Image (45°)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rotated_magnitude_spectrum, cmap='gray')
    axes[1, 1].set_title("Rotated Magnitude Spectrum")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")  
    plt.tight_layout()
    plt.show()

image_path = "l4a.jpeg"
process_fft(image_path)
