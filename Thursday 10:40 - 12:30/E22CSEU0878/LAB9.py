import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from skimage.metrics import peak_signal_noise_ratio as psnr

# ---------- IMAGE PROCESSING TASKS ----------

def sift_feature_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    output = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("SIFT Features", output)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

def feature_matching(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: Could not read one or both images.")
        return
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        print("Error: No keypoints detected in one or both images.")
        return
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("ORB Feature Matching", result)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

def contour_detection_watershed(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding using OTSU's method
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological transformations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform to find sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Identify unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label the connected components (sure foreground)
    markers = cv2.connectedComponents(sure_fg)[1]
    markers += 1
    markers[unknown == 255] = 0

    # Perform watershed algorithm
    markers = cv2.watershed(image, markers)

    # Mark boundaries
    image[markers == -1] = [0, 0, 255]

    # Show the result
    cv2.imshow("Watershed Contours", image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# ---------- AUTOENCODER IMAGE RESTORATION ----------

def autoencoder_denoising(num_epochs=10, batch_size=128, noise_factor=0.5, num_images=5):
    # Load and preprocess data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Add noise to the images
    def add_noise(imgs, noise_factor=0.5):
        noisy_imgs = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
        return np.clip(noisy_imgs, 0.0, 1.0)

    x_train_noisy = add_noise(x_train, noise_factor)
    x_test_noisy = add_noise(x_test, noise_factor)

    # Build the autoencoder model
    input_layer = Input(shape=(28, 28, 1))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Compile the model
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder
    autoencoder.fit(x_train_noisy, x_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test_noisy, x_test))

    # Denoise the test images
    x_test_denoised = autoencoder.predict(x_test_noisy)

    # Compute PSNR
    psnr_value = np.mean([tf.image.psnr(x_test[i], x_test_denoised[i], max_val=1.0) for i in range(len(x_test))])
    print(f"Average PSNR: {psnr_value:.2f} dB")

    # Plot the original, noisy, and denoised images
    def plot_images(original, noisy, denoised, num_images=5):
        plt.figure(figsize=(10, 5))
        for i in range(num_images):
            # Plot the original images
            plt.subplot(3, num_images, i + 1)
            plt.imshow(original[i].squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0: plt.title("Original")

            # Plot the noisy images
            plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(noisy[i].squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0: plt.title("Noisy")

            # Plot the denoised images
            denoised_image = denoised[i].squeeze()
            if denoised_image.max() <= 1:
                denoised_image = (denoised_image * 255).astype(np.uint8)
            plt.subplot(3, num_images, i + 1 + 2 * num_images)
            plt.imshow(denoised_image, cmap='gray')
            plt.axis('off')
            if i == 0: plt.title("Denoised")

        plt.tight_layout()
        plt.show()

    # Call the function to plot images
    plot_images(x_test, x_test_noisy, x_test_denoised, num_images)

# ---------- RUNNING EVERYTHING ----------
if __name__ == "__main__":
    # sift_feature_detection("sift_img.png")
    # feature_matching("img1_1.png", "img1_2.png")
    # contour_detection_watershed("contour_img1.png")
    autoencoder_denoising()
