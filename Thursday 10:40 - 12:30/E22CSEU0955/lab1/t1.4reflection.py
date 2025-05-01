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


def reflect(vertices, axis):
    if axis == 'x':
        transformation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        transformation_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif axis == 'origin':
        transformation_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    return (transformation_matrix @ vertices.T).T[:, :2]
reflected_triangle = reflect(triangle, 'x')
plot_object(reflected_triangle, "Reflected Object (x-axis)")
