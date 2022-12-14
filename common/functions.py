import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)                                    # 防止溢出
        x = np.exp(x)
        x = x / x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 将 one-hot 转换成正确解的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return loss
