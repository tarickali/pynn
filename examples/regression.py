"""Simple regression example using pynn."""

import numpy as np
from sklearn.datasets import make_regression

from pynn.core import Tensor
from pynn.nn import Linear, Sequential
from pynn.nn.losses import MeanSquaredError
from pynn.optim import SGD
from pynn.utils.data import get_batches


def regression():
    X, y = make_regression(n_samples=200, n_features=10)
    y = y.reshape(-1, 1)

    # Normalize data
    X = (X - X.mean()) / (X.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    model = Sequential(
        [
            Linear(10, 16, activation="relu"),
            Linear(16, 16, activation="relu"),
            Linear(16, 1),
        ]
    )

    loss_fn = MeanSquaredError()
    optimizer = SGD(model.parameters, learning_rate=0.01)

    EPOCHS = 300
    history = []
    for e in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for Xb, yb in get_batches(X, y, batch_size=32):
            X_t = Tensor(Xb)
            y_t = Tensor(yb)
            pred = model(X_t)
            loss_t = loss_fn(y_t, pred)
            epoch_loss += float(np.asarray(loss_t.data).flat[0])
            n_batches += 1

            model.zero_grad()
            loss_t.backward()
            optimizer.update()

        history.append(epoch_loss / n_batches)
        if (e + 1) % 50 == 0:
            print(f"Epoch {e + 1} -- loss {history[-1]:.6f}")

    # Final prediction
    pred = model(Tensor(X))
    pred_np = pred.data
    print("Sample | True | Pred")
    for i in range(min(5, len(y))):
        print(f"  {i}    | {y[i, 0]:.3f} | {pred_np[i, 0]:.3f}")


if __name__ == "__main__":
    regression()
