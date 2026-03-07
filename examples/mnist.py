"""MNIST classification example using pynn."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pynn.core import Tensor
from pynn.utils.data import one_hot, get_batches
from pynn.nn import Linear, Sequential
from pynn.nn.losses import CategoricalCrossentropy
from pynn.nn.activations import Softmax
from pynn.optim import SGD


def generate_data(data_path: str = "examples/data/mnist/train.csv"):
    """Load and preprocess MNIST data from CSV. Returns (X, y, input_dim)."""
    try:
        train_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"MNIST data not found at {data_path}. Using synthetic data for demo.")
        np.random.seed(42)
        # Low-dim synthetic task: easy to learn so the demo shows clear progress
        input_dim = 50
        n_samples = 1000
        X = np.random.randn(n_samples, input_dim).astype(np.float64) * 0.5
        W_fake = np.random.randn(input_dim, 10).astype(np.float64)
        logits = X @ W_fake
        y_idx = np.argmax(logits, axis=1)
        y = one_hot(y_idx, 10)
        return X, y, input_dim

    pixels = train_df.drop("label", axis=1).to_numpy()
    labels = train_df["label"].to_numpy()
    X = pixels.reshape((-1, 784)).astype(np.float64) / 255.0
    y = one_hot(labels, 10)
    return X, y, 784


def mnist_driver():
    X, y, input_dim = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    BATCH_SIZE = 32
    N_CLASSES = 10
    if input_dim == 784:
        ALPHA = 0.01
        EPOCHS = 15
    else:
        ALPHA = 0.05
        EPOCHS = 40

    # Model size depends on data: small net for synthetic demo, larger for real MNIST
    if input_dim == 784:
        model = Sequential(
            [
                Linear(784, 256, activation="relu"),
                Linear(256, 128, activation="relu"),
                Linear(128, N_CLASSES, activation="identity"),
            ]
        )
    else:
        model = Sequential(
            [
                Linear(input_dim, 64, activation="relu"),
                Linear(64, N_CLASSES, activation="identity"),
            ]
        )

    loss_fn = CategoricalCrossentropy(logits=True)
    optimizer = SGD(model.parameters, learning_rate=ALPHA, momentum=0.9, dampening=0.1)
    softmax = Softmax()

    n_train = X_train.shape[0]
    history = []
    for e in range(EPOCHS):
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0
        for Xb, yb in get_batches(X_train, y_train, batch_size=BATCH_SIZE):
            X_t = Tensor(Xb)
            y_t = Tensor(yb)
            pred = model(X_t)
            loss_t = loss_fn(y_t, pred)
            epoch_loss += float(np.asarray(loss_t.data).flat[0])
            n_batches += 1

            model.zero_grad()
            loss_t.backward()
            optimizer.update()

            prob = softmax(pred).data
            pred_labels = np.argmax(prob, axis=1)
            true_labels = np.argmax(yb, axis=1)
            epoch_correct += np.sum(pred_labels == true_labels)

        avg_loss = epoch_loss / n_batches
        acc = epoch_correct / n_train
        history.append({"epoch": e + 1, "loss": avg_loss, "acc": acc})
        print(f"Epoch {e + 1} -- loss {avg_loss:.4f} -- acc {acc:.4f}")

    # Train accuracy
    pred_train = model(Tensor(X_train))
    prob_train = softmax(pred_train).data
    train_pred_labels = np.argmax(prob_train, axis=1)
    train_true = np.argmax(y_train, axis=1)
    train_acc = np.mean(train_pred_labels == train_true)
    print(f"Train accuracy: {train_acc:.4f}")

    # Test accuracy
    pred_test = model(Tensor(X_test))
    prob_test = softmax(pred_test).data
    test_pred_labels = np.argmax(prob_test, axis=1)
    test_true = np.argmax(y_test, axis=1)
    test_acc = np.mean(test_pred_labels == test_true)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    mnist_driver()
