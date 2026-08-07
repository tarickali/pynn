#!/usr/bin/env bash
# Run the GitHub Actions matrix locally, step for step, on every supported Python.
#
#     scripts/ci_matrix.sh
#
# Three CI failures in a row were invisible on the newest interpreter: mypy aborting
# on older NumPy stubs, argparse rejecting an empty nargs="*" before 3.12, and
# annotation faults that only the newest NumPy stubs catch. Waiting on a push to find
# out is a slow loop; this closes it.
#
# Each version gets a throwaway virtualenv under $ENVS, reused on later runs. Needs the
# interpreters on PATH as python3.10 ... python3.14 (`brew install python@3.13`, etc.);
# versions that are missing are reported and skipped.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS="${CI_MATRIX_ENVS:-${TMPDIR:-/tmp}/pynn-ci-matrix}"
SCRATCH="$ENVS/logs"
mkdir -p "$SCRATCH"
cd "$REPO" || exit 1
echo "environments: $ENVS"
echo

for V in 3.10 3.11 3.12 3.13 3.14; do
  PY="$(command -v "python$V" || true)"
  ENV="$ENVS/ci$V"
  echo "############ Python $V ############"
  if [ -z "$PY" ]; then
    echo "  python$V not on PATH — skipped"; echo; continue
  fi
  if [ ! -x "$ENV/bin/python" ]; then
    rm -rf "$ENV"
    "$PY" -m venv "$ENV" || { echo "  venv FAILED"; echo; continue; }
  fi
  "$ENV/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1
  echo -n "  Install:        "
  if "$ENV/bin/python" -m pip install -q -e ".[dev]" >"$SCRATCH/install-$V.log" 2>&1; then
    echo "ok  (numpy $("$ENV/bin/python" -c 'import numpy;print(numpy.__version__)' 2>/dev/null))"
  else
    echo "FAILED — see install-$V.log"; tail -5 "$SCRATCH/install-$V.log"; continue
  fi

  echo -n "  Ruff check:     "
  "$ENV/bin/ruff" check pynn tests examples scripts benchmarks >"$SCRATCH/ruff-$V.log" 2>&1 \
    && echo "ok" || { echo "FAILED (exit $?)"; tail -12 "$SCRATCH/ruff-$V.log"; }

  echo -n "  Ruff format:    "
  "$ENV/bin/ruff" format --check pynn tests examples scripts benchmarks >"$SCRATCH/fmt-$V.log" 2>&1 \
    && echo "ok" || { echo "FAILED (exit $?)"; tail -12 "$SCRATCH/fmt-$V.log"; }

  echo -n "  Mypy:           "
  "$ENV/bin/mypy" >"$SCRATCH/mypy-$V.log" 2>&1 \
    && echo "ok" || { echo "FAILED (exit $?)"; tail -12 "$SCRATCH/mypy-$V.log"; }

  echo -n "  Pytest+cov:     "
  "$ENV/bin/python" -m pytest -m "not external" --cov=pynn --cov-branch \
      --cov-report=xml --cov-report=term-missing -q >"$SCRATCH/pytest-$V.log" 2>&1 \
    && echo "ok  ($(grep -oE '[0-9]+ passed[^,]*' "$SCRATCH/pytest-$V.log" | tail -1))" \
    || { echo "FAILED (exit $?)"; tail -25 "$SCRATCH/pytest-$V.log"; }

  echo -n "  Verify:         "
  "$ENV/bin/python" -m pynn.verify >"$SCRATCH/verify-$V.log" 2>&1 \
    && echo "ok  ($(tail -1 "$SCRATCH/verify-$V.log"))" \
    || { echo "FAILED (exit $?)"; tail -15 "$SCRATCH/verify-$V.log"; }
  echo
done
echo "############ done ############"
