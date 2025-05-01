import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

image = cv2.imread('lenna.png')  
if image is None:
    print("Image not found. Please check the file path.")
    exit()

display_image(image, "Original Image")

# 1 TRANSLATION
def translate(image, tx, ty):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

translated_image = translate(image, 50, 80)
display_image(translated_image, "Translated Image")

# 2 SCALING
def scale(image, sx, sy):
    _, _ = image.shape[:2] 
    scaled_image = cv2.resize(image, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    return scaled_image

scaled_image = scale(image, 5, 3)
display_image(scaled_image, "Scaled Image")

# 3 ROTATION
def rotate(image, angle):
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

rotated_image = rotate(image, 45)
display_image(rotated_image, "Rotated Image")

# 4 REFLECTION
def reflect(image, axis):
    if axis == 'x':
        reflected_image = cv2.flip(image, 0)
    elif axis == 'y':
        reflected_image = cv2.flip(image, 1)
    else:
        raise ValueError("Invalid axis. Use 'x'or 'y'")
    return reflected_image

reflected_image = reflect(image, 'y')
display_image(reflected_image, "Reflected Image")

# 5 SHEARING
def shear(image, shx, shy):
    rows, cols = image.shape[:2]
    shear_matrix = np.float32([
        [1, shx, 0],
        [shy, 1, 0]
    ])
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols + int(abs(shx * rows)), rows + int(abs(shy * cols))))
    return sheared_image

sheared_image_x = shear(image, 0.5, 0)
sheared_image_y = shear(image, 0, 0.5)
display_image(sheared_image_x, "Sheared Image (x-axis)")
display_image(sheared_image_y, "Sheared Image (y-axis)")

# 6 CROPPING
def crop(image, x, y, w, h):
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

cropped_image = crop(image, 100, 100, 200, 200)
display_image(cropped_image, "Cropped Image")
