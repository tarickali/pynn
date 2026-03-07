import os
import sys


def main():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        print("Install dependencies: pip install numpy pandas scikit-learn")
        sys.exit(1)

    print("Fetching MNIST from openml...")
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)

    cols = ["label"] + [f"pixel{i}" for i in range(784)]
    df = pd.DataFrame(np.hstack([y.reshape(-1, 1), X]), columns=cols)
    df["label"] = df["label"].astype(int)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "examples", "data", "mnist"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train.csv")

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")
    print("Run: py3 -m examples.mnist")


if __name__ == "__main__":
    main()
