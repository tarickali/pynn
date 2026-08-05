import sys

import numpy as np

# Add project root so "pynn" package is importable
sys.path.insert(0, ".")


def main():
    from pynn.core import Tensor
    from pynn.nn import Linear, Sequential
    from pynn.nn.losses import MeanSquaredError
    from pynn.optim import SGD

    rng = np.random.default_rng(42)
    X = rng.standard_normal((8, 4))
    y = rng.standard_normal((8, 1))

    model = Sequential(
        [
            Linear(4, 8, activation="relu"),
            Linear(8, 1),
        ]
    )
    loss_fn = MeanSquaredError()
    optimizer = SGD(model.parameters, lr=0.01)

    X_t = Tensor(X)
    y_t = Tensor(y)
    pred = model(X_t)
    loss = loss_fn(y_t, pred)
    loss_val_before = float(np.asarray(loss.data).flat[0])

    model.zero_grad()
    loss.backward()
    optimizer.update()

    pred2 = model(X_t)
    loss2 = loss_fn(y_t, pred2)
    loss_val_after = float(np.asarray(loss2.data).flat[0])

    assert loss_val_after <= loss_val_before + 1e-5, (
        "Loss should decrease or stay similar after a step"
    )
    print("Smoke test passed: forward, backward, and optimizer step OK.")


if __name__ == "__main__":
    main()
