"""Binary classification example (circles/moons) using pynn."""

import numpy as np
from sklearn.datasets import make_circles

from pynn.core import Tensor
from pynn.nn import Linear, Sequential
from pynn.nn.losses import BinaryCrossentropy
from pynn.optim import SGD


def binary_classification():
    X, y = make_circles(256, random_state=42)
    y = y.reshape(-1, 1).astype(np.float64)

    ALPHA = 0.1
    EPOCHS = 1000

    model = Sequential(
        [
            Linear(2, 64, activation="relu"),
            Linear(64, 64, activation="relu"),
            Linear(64, 1, activation="sigmoid"),
        ]
    )

    loss_fn = BinaryCrossentropy(logits=False)
    optimizer = SGD(model.parameters, learning_rate=ALPHA, momentum=0.9)

    history = []
    for e in range(EPOCHS):
        X_t = Tensor(X)
        y_t = Tensor(y)
        pred = model(X_t)
        loss_t = loss_fn(y_t, pred)
        loss_val = float(np.asarray(loss_t.data).flat[0])
        history.append(loss_val)

        model.zero_grad()
        loss_t.backward()
        optimizer.update()

        if (e + 1) % 100 == 0:
            print(f"Epoch {e + 1} -- loss {loss_val:.6f}")

    pred = model(Tensor(X))
    pred_np = (pred.data >= 0.5).astype(np.float64)
    acc = np.mean(pred_np == y)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    binary_classification()
