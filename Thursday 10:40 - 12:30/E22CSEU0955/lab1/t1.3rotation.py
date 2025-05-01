import numpy as np
import matplotlib.pyplot as plt
def plot_object(vertices, title):
    vertices = np.append(vertices, [vertices[0]], axis=0)
    plt.plot(vertices[:, 0], vertices[:, 1], marker='o')
    plt.title(title)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

triangle = np.array([
    [0, 0],
    [1, 0],
    [0.5, 1]
])

plot_object(triangle, "Original Object")


def rotate(vertices, angle):
    rad = np.radians(angle)
    transformation_matrix = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    return (transformation_matrix @ vertices.T).T[:, :2]
rotated_triangle = rotate(triangle, 45)
plot_object(rotated_triangle, "Rotated Object")
