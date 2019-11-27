import numpy as np


# Gaussian 2D mixture
# Inspired from https://github.com/hwalsuklee/tensorflow-mnist-AAE

def transform_sample(x, y, label, num_labels, shift=1.4):
    radius = 2.0 * np.pi / float(num_labels) * float(label)
    # rotate
    rot_x = x * np.cos(radius) - y * np.sin(radius)
    rot_y = x * np.sin(radius) + y * np.cos(radius)
    # shift
    x = rot_x + (shift * np.cos(radius))
    y = rot_y + (shift * np.sin(radius))
    return np.array([x, y]).reshape((2, ))


def gaussian_mixture(batch_size, num_labels, x_std, y_std, labels):
    if labels is None:
        labels = np.random.randint(0, num_labels, batch_size)

    x = np.random.normal(0, x_std, (batch_size, 1))
    y = np.random.normal(0, y_std, (batch_size, 1))
    z = np.empty((batch_size, 2), dtype=np.float32)
    for i in range(batch_size):
        z[i, :] = transform_sample(x[i], y[i], labels[i], num_labels)
    return z
