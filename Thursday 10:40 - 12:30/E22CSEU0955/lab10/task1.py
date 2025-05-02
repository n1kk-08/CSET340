import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise
import os
import matplotlib.pyplot as plt

img = img_as_float(io.imread('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab10/a.jpg', as_gray=True))
noisy_img = random_noise(img, var=0.01)

median_denoised = cv2.medianBlur((noisy_img * 255).astype(np.uint8), 3)
wavelet_denoised = denoise_wavelet(noisy_img, channel_axis=None)

def compare_metrics(original, denoised, name):
    print(f"{name} PSNR: {psnr(original, denoised, data_range=1.0):.2f}, "
          f"SSIM: {ssim(original, denoised, data_range=1.0):.2f}, "
          f"MSE: {mse(original, denoised):.5f}")

compare_metrics(img, median_denoised / 255.0, "Median")
compare_metrics(img, wavelet_denoised, "Wavelet")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(noisy_img, cmap='gray')
axs[1].set_title('Noisy Image')
axs[1].axis('off')

axs[2].imshow(median_denoised / 255.0, cmap='gray')
axs[2].set_title('Median Denoised')
axs[2].axis('off')

axs[3].imshow(wavelet_denoised, cmap='gray')
axs[3].set_title('Wavelet Denoised')
axs[3].axis('off')

plt.tight_layout()
plt.show()

cap = cv2.VideoCapture('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab10/video.mp4')
os.makedirs('frames', exist_ok=True)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f'frames/frame_{frame_id:04d}.jpg', frame)
    frame_id += 1

cap.release()
print(f"Total frames extracted: {frame_id}")

cap = cv2.VideoCapture('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab10/video.mp4')
os.makedirs('processed_frames', exist_ok=True)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    bitwise_frame = cv2.bitwise_not(gray)
    gaussian_frame = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_frame = cv2.Canny(gray, 100, 200)

    cv2.imwrite(f'processed_frames/frame_{frame_id:04d}_adaptive_thresh.jpg', adaptive_thresh)
    cv2.imwrite(f'processed_frames/frame_{frame_id:04d}_bitwise.jpg', bitwise_frame)
    cv2.imwrite(f'processed_frames/frame_{frame_id:04d}_gaussian.jpg', gaussian_frame)
    cv2.imwrite(f'processed_frames/frame_{frame_id:04d}_canny.jpg', canny_frame)
    frame_id += 1

cap.release()

frame_to_display = cv2.imread('processed_frames/frame_0000_canny.jpg')

if frame_to_display is not None:
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].imshow(cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Frame')
    axs[0, 0].axis('off')

    adaptive_thresh = cv2.imread('processed_frames/frame_0000_adaptive_thresh.jpg', cv2.IMREAD_GRAYSCALE)
    axs[0, 1].imshow(adaptive_thresh, cmap='gray')
    axs[0, 1].set_title('Adaptive Threshold')
    axs[0, 1].axis('off')

    bitwise_frame = cv2.imread('processed_frames/frame_0000_bitwise.jpg', cv2.IMREAD_GRAYSCALE)
    axs[0, 2].imshow(bitwise_frame, cmap='gray')
    axs[0, 2].set_title('Bitwise Frame')
    axs[0, 2].axis('off')

    gaussian_frame = cv2.imread('processed_frames/frame_0000_gaussian.jpg', cv2.IMREAD_GRAYSCALE)
    axs[1, 0].imshow(gaussian_frame, cmap='gray')
    axs[1, 0].set_title('Gaussian Frame')
    axs[1, 0].axis('off')

    canny_frame = cv2.imread('processed_frames/frame_0000_canny.jpg', cv2.IMREAD_GRAYSCALE)
    axs[1, 1].imshow(canny_frame, cmap='gray')
    axs[1, 1].set_title('Canny Frame')
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not load the frame for display.")