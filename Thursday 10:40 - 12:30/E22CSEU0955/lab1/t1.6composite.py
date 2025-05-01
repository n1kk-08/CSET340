import numpy as np
import matplotlib.pyplot as plt

def plot_object(vertices, title):
    vertices = np.append(vertices, [vertices[0]], axis=0)  
    plt.plot(vertices[:, 0], vertices[:, 1], marker='o')
    plt.title(title)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def composite_transform(vertices, transformations):
    composite_matrix = np.eye(3)
    for matrix in transformations:
        composite_matrix = matrix @ composite_matrix
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    return (composite_matrix @ vertices.T).T[:, :2]

triangle = np.array([
    [0, 0],
    [1, 0],
    [0.5, 1]
])

plot_object(triangle, "Original Object")

translation_matrix = np.array([
    [1, 0, 2],
    [0, 1, 3],
    [0, 0, 1]
])
rotation_matrix = np.array([
    [np.cos(np.radians(45)), -np.sin(np.radians(45)), 0],
    [np.sin(np.radians(45)), np.cos(np.radians(45)), 0],
    [0, 0, 1]
])
composite_triangle = composite_transform(triangle, [translation_matrix, rotation_matrix])
plot_object(composite_triangle, "Composite Transformation (Translation + Rotation)")
