import cv2
import matplotlib.pyplot as plt

def compute_histogram(image, mask=None):
    hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
    return hist

def plot_histogram_subplot(ax, image, mode="count"):
    if len(image.shape) == 2:  
        hist = compute_histogram(image)
        ax.plot(hist, color='black')
    else:  
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    
    ax.set_xlabel("Intensity Values")
    ax.set_ylabel("Probability" if mode == "probability" else "Pixel Count")

def equalize_grayscale(image):
    return cv2.equalizeHist(image)

def process_image(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = equalize_grayscale(grayscale)

    _, axes = plt.subplots(3, 2, figsize=(10, 10))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Colored")
    axes[0, 0].axis("off")
    plot_histogram_subplot(axes[0, 1], image, "Original Color")

    axes[1, 0].imshow(grayscale, cmap='gray')
    axes[1, 0].set_title("Original Grayscale")
    axes[1, 0].axis("off")
    plot_histogram_subplot(axes[1, 1], grayscale, "Grayscale")

    axes[2, 0].imshow(equalized, cmap='gray')
    axes[2, 0].set_title("Equalized Grayscale")
    axes[2, 0].axis("off")
    plot_histogram_subplot(axes[2, 1], equalized, "Equalized Grayscale")

    plt.tight_layout()
    plt.show()

image_path = "l4b.jpg"
process_image(image_path)
