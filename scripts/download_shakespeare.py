"""Fetch the Tiny Shakespeare corpus for `examples/char_rnn.ipynb`.

Stdlib only, unlike `download_mnist.py`: the corpus is one plain text file over HTTP,
so there is nothing here that `urllib` does not already do, and the notebook stays
runnable without the `[mnist]` extra.
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Andrej Karpathy's char-rnn corpus: ~1.1 MB of Shakespeare, the conventional
#: character-level language modelling benchmark. The plays themselves are public
#: domain; this is the concatenation everyone compares against.
URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

#: Below this the response is an error page or a truncated transfer rather than the
#: corpus, and the notebook would train on it without complaining.
MINIMUM_BYTES = 1_000_000


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "examples/data/shakespeare"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "input.txt"

    if out_file.exists():
        print(f"Already present: {out_file} ({out_file.stat().st_size:,} bytes)")
        return

    print(f"Fetching {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Download failed: {error}")
        sys.exit(1)

    if len(payload) < MINIMUM_BYTES:
        print(f"Got {len(payload):,} bytes, expected at least {MINIMUM_BYTES:,}.")
        sys.exit(1)

    text = payload.decode("utf-8")
    out_file.write_text(text, encoding="utf-8")

    print(f"Saved {len(text):,} characters to {out_file}")
    print(f"Vocabulary: {len(set(text))} distinct characters")
    print("Run: jupyter lab examples/char_rnn.ipynb")


if __name__ == "__main__":
    main()
