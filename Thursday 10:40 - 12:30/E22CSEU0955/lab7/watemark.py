import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_watermark(image, watermark_text='Watermark', position=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, watermark_text, position, font, font_scale, color, thickness)
    return image

image_info = [
    ('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab7/a.png', 'Blood Cell Image'),
    ('C:/Users/cchan/OneDrive/Documents/COLLEGE/SEMESTERS/Semester 6/340 l/lab/lab7/b.png', 'Satellite Image')
]

for image_path, title in image_info:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    image_with_watermark = add_watermark(image.copy(), 'Block 1')
    axes[0, 0].imshow(image_with_watermark, cmap='gray')

    log = cv2.Laplacian(image, cv2.CV_64F)
    axes[0, 1].imshow(log, cmap='gray')
    axes[0, 1].set_title("LoG Blob Detection")
    axes[0, 1].axis('off')
    log_with_watermark = add_watermark(log.copy(), 'Block 2')
    axes[0, 1].imshow(log_with_watermark, cmap='gray')

    blur1 = cv2.GaussianBlur(image, (5, 5), 1)
    blur2 = cv2.GaussianBlur(image, (5, 5), 2)
    dog = blur1 - blur2
    axes[0, 2].imshow(dog, cmap='gray')
    axes[0, 2].set_title("DoG Blob Detection")
    axes[0, 2].axis('off')
    dog_with_watermark = add_watermark(dog.copy(), 'Block 3')
    axes[0, 2].imshow(dog_with_watermark, cmap='gray')

    alpha = 1.5
    beta = 20
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    axes[1, 0].imshow(enhanced, cmap='gray')
    axes[1, 0].set_title("Brightness & Contrast Adjustment")
    axes[1, 0].axis('off')
    enhanced_with_watermark = add_watermark(enhanced.copy(), 'Block 4')
    axes[1, 0].imshow(enhanced_with_watermark, cmap='gray')

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    axes[1, 1].imshow(sharpened, cmap='gray')
    axes[1, 1].set_title("Sharpened Image")
    axes[1, 1].axis('off')
    sharpened_with_watermark = add_watermark(sharpened.copy(), 'Block 5')
    axes[1, 1].imshow(sharpened_with_watermark, cmap='gray')

    equalized = cv2.equalizeHist(image)
    axes[1, 2].imshow(equalized, cmap='gray')
    axes[1, 2].set_title("Histogram Equalization")
    axes[1, 2].axis('off')
    equalized_with_watermark = add_watermark(equalized.copy(), 'Block 6')
    axes[1, 2].imshow(equalized_with_watermark, cmap='gray')

    plt.tight_layout()
    plt.show()
