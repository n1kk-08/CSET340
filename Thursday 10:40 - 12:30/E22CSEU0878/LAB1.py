import numpy as np
import matplotlib.pyplot as plt

def transform(vertices, matrix):
    vertices_h = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  
    return (vertices_h @ matrix.T)[:, :2]  

def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

def scaling_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

def rotation_matrix(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                     [np.sin(angle_rad),  np.cos(angle_rad), 0],
                     [0, 0, 1]])

def reflection_matrix(axis):
    if axis == 'x':
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'y':
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis == 'origin':
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

def shearing_matrix(shx, shy):
    return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])

def plot_shape(vertices, label, color='blue'):
    vertices = np.vstack((vertices, vertices[0]))  
    plt.plot(vertices[:, 0], vertices[:, 1], marker='o', label=label, color=color)

if __name__ == "__main__":
    vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

    plt.figure(figsize=(8, 6))
    plot_shape(vertices, "Original", "blue")

    translated = transform(vertices, translation_matrix(3, 2))
    scaled = transform(vertices, scaling_matrix(1.5, 0.5))
    rotated = transform(vertices, rotation_matrix(45))
    reflected = transform(vertices, reflection_matrix('x'))
    sheared = transform(vertices, shearing_matrix(1, 0.5))

    plot_shape(translated, "Translated (3,2)", "green")
    plot_shape(scaled, "Scaled (1.5,0.5)", "orange")
    plot_shape(rotated, "Rotated (45°)", "red")
    plot_shape(reflected, "Reflected (x-axis)", "purple")
    plot_shape(sheared, "Sheared (1,0.5)", "brown")

    composite = transform(vertices, translation_matrix(2, 1) @ scaling_matrix(1.2, 1.2))
    plot_shape(composite, "Composite (Translate + Scale)", "cyan")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("2D Transformations")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

import cv2
import numpy as np

# Load the images
color_img = cv2.imread('Colour_Img.PNG')
gray_img = cv2.imread('GrayScale.PNG', 0)

def show_image(title, img, wait_time=3000):
    cv2.imshow(title, img)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

# 1. Image Translation
def translate_image(img, tx, ty):
    rows, cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, translation_matrix, (cols, rows))

# 2. Reflection
def reflect_image(img, axis):
    if axis == 'x':
        return cv2.flip(img, 0)
    elif axis == 'y':
        return cv2.flip(img, 1)

# 3. Rotation
def rotate_image(img, angle):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))

# 4. Scaling
def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

# 5. Cropping
def crop_image(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

# 6. Shearing in X-axis
def shear_image_x(img, shear_factor):
    rows, cols = img.shape[:2]
    shearing_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(img, shearing_matrix, (cols, rows))

# 7. Shearing in Y-axis
def shear_image_y(img, shear_factor):
    rows, cols = img.shape[:2]
    shearing_matrix = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    return cv2.warpAffine(img, shearing_matrix, (cols, rows))

# Example transformations
translated = translate_image(color_img, 50, 50)
rotated = rotate_image(color_img, 45)
scaled = scale_image(color_img, 0.5, 0.5)
cropped = crop_image(color_img, 50, 50, 200, 200)
sheared_x = shear_image_x(color_img, 0.5)
sheared_y = shear_image_y(color_img, 0.5)
reflected = reflect_image(color_img, 'x')

# Display results
show_image("Translated Image", translated)
show_image("Rotated Image", rotated)
show_image("Scaled Image", scaled)
show_image("Cropped Image", cropped)
show_image("Sheared X Image", sheared_x)
show_image("Sheared Y Image", sheared_y)
show_image("Reflected Image", reflected)
