# # meta_learning_mnist.py
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.metrics import classification_report
# from scipy.ndimage import map_coordinates, gaussian_filter

# # Load and preprocess MNIST
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.astype(np.float32) / 255.0
# x_test = x_test.astype(np.float32) / 255.0
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# # Elastic deformation (applied only on training data)
# def elastic_transform(image, alpha=36, sigma=6):
#     random_state = np.random.RandomState(None)
#     shape = image.shape[:2]
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
#     x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#     indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
#     distorted = map_coordinates(image[:, :, 0], indices, order=1, mode='reflect')
#     return np.reshape(distorted, image.shape)

# x_train_aug = np.array([elastic_transform(img) for img in x_train])
# x_train = np.expand_dims(x_train_aug, -1)

# # Group images by class
# train_classes = {i: x_train[y_train == i] for i in range(10)}
# test_classes = {i: x_test[y_test == i] for i in range(10)}

# # CNN encoder
# def create_encoder():
#     model = models.Sequential([
#         layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, 3, activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(64)
#     ])
#     return model

# # Few-shot sampler
# def sample_episode(classes, n_classes=5, k_shot=5, q_query=5):
#     selected = np.random.choice(list(classes.keys()), n_classes, replace=False)
#     support_x, support_y, query_x, query_y = [], [], [], []
#     for idx, cls in enumerate(selected):
#         samples = np.random.choice(len(classes[cls]), k_shot + q_query, replace=False)
#         support = classes[cls][samples[:k_shot]]
#         query = classes[cls][samples[k_shot:]]
#         support_x.append(support)
#         query_x.append(query)
#         support_y += [idx] * k_shot
#         query_y += [idx] * q_query
#     return np.concatenate(support_x), np.array(support_y), np.concatenate(query_x), np.array(query_y)

# # Prototypes
# def compute_prototypes(embeddings, labels, N, K):
#     return np.array([np.mean(embeddings[labels == i], axis=0) for i in range(N)])

# def prototypical_loss(prototypes, query_embeddings, query_labels):
#     dists = tf.norm(tf.expand_dims(query_embeddings, 1) - prototypes, axis=2)
#     preds = tf.argmin(dists, axis=1)
#     acc = tf.reduce_mean(tf.cast(preds == query_labels, tf.float32))
#     loss = tf.reduce_mean((tf.norm(query_embeddings - tf.gather(prototypes, query_labels), axis=1))**2)
#     return loss, acc

# # Train Prototypical Network
# def train_prototypical(model, classes, episodes=1000, N=5, K=5, Q=5):
#     optimizer = tf.keras.optimizers.Adam(1e-3)
#     for ep in range(episodes):
#         sx, sy, qx, qy = sample_episode(classes, N, K, Q)
#         with tf.GradientTape() as tape:
#             s_embed = model(sx, training=True)
#             q_embed = model(qx, training=True)
#             prototypes = compute_prototypes(s_embed.numpy(), sy, N, K)
#             prototypes = tf.convert_to_tensor(prototypes, dtype=tf.float32)
#             loss, acc = prototypical_loss(prototypes, q_embed, tf.convert_to_tensor(qy))
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         if ep % 100 == 0:
#             print(f"Episode {ep}, Loss: {loss:.4f}, Acc: {acc:.4f}")

# # Evaluate on test classes
# def evaluate_prototypical(model, classes, episodes=100, N=5, K=5, Q=5):
#     accs = []
#     for _ in range(episodes):
#         sx, sy, qx, qy = sample_episode(classes, N, K, Q)
#         s_embed = model(sx, training=False)
#         q_embed = model(qx, training=False)
#         prototypes = compute_prototypes(s_embed.numpy(), sy, N, K)
#         prototypes = tf.convert_to_tensor(prototypes, dtype=tf.float32)
#         _, acc = prototypical_loss(prototypes, q_embed, tf.convert_to_tensor(qy))
#         accs.append(acc.numpy())
#     print(f"Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# # MAIN
# proto_model = create_encoder()
# train_prototypical(proto_model, train_classes)
# evaluate_prototypical(proto_model, test_classes)



import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Load and preprocess MNIST
def load_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

# Encoder model (ConvNet)
def get_encoder():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # Normalize embeddings
    ])
    return model

# Create episodic batches
def create_episode(x, y, N=5, K=1, Q=5):
    classes = np.random.choice(np.unique(y), N, replace=False)
    support_x, support_y, query_x, query_y = [], [], [], []
    for i, cls in enumerate(classes):
        cls_idx = np.where(y == cls)[0]
        selected = np.random.choice(cls_idx, K + Q, replace=False)
        support_x.extend(x[selected[:K]])
        support_y.extend([i] * K)
        query_x.extend(x[selected[K:]])
        query_y.extend([i] * Q)
    return (np.array(support_x), np.array(support_y),
            np.array(query_x), np.array(query_y))

# Compute class prototypes
def compute_prototypes(embeddings, labels, N, K):
    return np.array([np.mean(embeddings[labels == i], axis=0) for i in range(N)])

# Prototypical loss and accuracy
def prototypical_loss(prototypes, query_embeddings, query_labels):
    distances = tf.norm(tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0), axis=-1)
    preds = tf.argmin(distances, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, query_labels), tf.float32))
    loss = tf.reduce_mean(tf.square(tf.gather_nd(distances, tf.stack([tf.range(len(query_labels)), query_labels], axis=1))))
    return loss, acc

# Training loop
def train_prototypical(model, x, y, episodes=1000, N=5, K=1, Q=5):
    optimizer = tf.keras.optimizers.Adam(1e-3)
    for episode in range(episodes):
        sx, sy, qx, qy = create_episode(x, y, N, K, Q)
        with tf.GradientTape() as tape:
            s_embed = model(sx, training=True)
            q_embed = model(qx, training=True)
            prototypes = compute_prototypes(s_embed, sy, N, K)
            loss, acc = prototypical_loss(prototypes, q_embed, qy)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}, Acc: {acc:.4f}")

# Evaluate trained model on a one-shot task
def evaluate_one_shot(model, x, y, N=5, K=1, Q=10):
    sx, sy, qx, qy = create_episode(x, y, N, K, Q)
    s_embed = model(sx, training=False)
    q_embed = model(qx, training=False)
    prototypes = compute_prototypes(s_embed, sy, N, K)
    distances = tf.norm(tf.expand_dims(q_embed, 1) - tf.expand_dims(prototypes, 0), axis=-1)
    preds = tf.argmin(distances, axis=1).numpy()
    print("\nEvaluation Report:")
    print(classification_report(qy, preds))

# Run everything
(x_train, y_train), (x_test, y_test) = load_preprocess_mnist()
encoder = get_encoder()
print("Training Prototypical Network...")
train_prototypical(encoder, x_train, y_train, episodes=1000, N=5, K=1, Q=5)
evaluate_one_shot(encoder, x_test, y_test, N=5, K=1, Q=10)
