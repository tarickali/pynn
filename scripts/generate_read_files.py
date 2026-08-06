"""Regenerate READ_FILES.md from the git history.

    python scripts/generate_read_files.py

A hand-maintained changelog of which commit touched which file would be stale within
a week. This reads it back out of git instead, so the file is current as of the last
time it was run.

Run it *after* committing, and commit the result on its own. Amending the commit it
was generated from would change that commit's hash and leave a hash in this file that
no longer exists, so the last commit listed is always the one before the refresh.

Commits are grouped into eras — stretches of work with a common purpose — declared in
`ERAS` below. Each entry names the commit the era starts at; everything from there up to
the next era belongs to it. Add a row when a new stretch of work begins.
"""

from __future__ import annotations

import collections
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = REPO / "READ_FILES.md"

#: (starting commit, title, description). Order matters; earliest first.
ERAS: list[tuple[str, str, str]] = [
    (
        "e2e7d87",
        "Era 1 — the original library",
        "The library as first written: the Tensor wrapper, the closure-based tape, "
        "activations, losses, and the first layers.",
    ),
    (
        "f7e7aa9",
        "Era 2 — first refactor",
        "The package layout settles into `core` / `functional` / `nn` / `optim` / "
        "`utils`; optimizers and packaging metadata arrive.",
    ),
    (
        "b904fa8",
        "Era 3 — correctness pass",
        "Every Tier 1 bug from PROJECT_REVIEW.md, plus the `pynn.verify` suite, pinned "
        "lint and type configuration, the im2col `conv2d`, and CI.",
    ),
    (
        "2c3e1d1",
        "Era 4 — structure, layers, and presentation",
        "The module tree, the new layers, `no_grad`, coverage to 98%, and the docs: "
        "DESIGN.md, the benchmarks, and the MNIST notebook.",
    ),
    (
        "742f92a",
        "Era 5 — environment and packaging",
        "Making the project reproducible from a clean clone: dependency groups, the "
        "CI matrix, and the documentation of how to run each part.",
    ),
]

STATUS = {"A": "added", "M": "modified", "D": "deleted"}
#: Files below this many commits are omitted from the frequency table — the long tail of
#: files touched once or twice says nothing about where the design moved.
FREQUENCY_FLOOR = 3

READING_ORDER = [
    (
        "pynn/core/tensor.py",
        "the tape: `add_children`, the `reverse` property, `backward`, the iterative "
        "topological sort",
    ),
    (
        "pynn/core/utils.py",
        "`unbroadcast` and `matrix_multiply_gradients` — the backward pass's shape "
        "plumbing",
    ),
    ("pynn/functional/activations.py", "the closure pattern, nine times over"),
    ("pynn/functional/losses.py", "why softmax and cross-entropy are fused"),
    ("pynn/core/module.py", "the module tree and every recursive walk over it"),
    (
        "pynn/nn/models.py",
        "`Sequential` as a `Module`, which is what lets containers nest",
    ),
    (
        "pynn/core/optimizer.py",
        "how parameter groups reach an optimizer, and why they stay live",
    ),
    (
        "pynn/functional/modules.py",
        "`conv2d`, the normalization layers, pooling",
    ),
    ("pynn/utils/array.py", "`im2col` / `col2im`"),
    ("pynn/core/grad_mode.py", "the two gates that turn recording off"),
    ("pynn/verify/gradients.py", "what is checked, and the shapes the checks take"),
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def anchor(title: str) -> str:
    """GitHub's heading-anchor rule: lowercase, drop punctuation, spaces to hyphens.

    An em dash surrounded by spaces leaves a double hyphen, which is why this cannot be
    a naive `replace(" ", "-")`.
    """
    kept = "".join(c for c in title.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


def status_label(code: str) -> str:
    return "renamed" if code.startswith("R") else STATUS.get(code[0], code)


def collect() -> list[dict]:
    commits = []
    for sha in git("log", "--reverse", "--format=%H").split():
        short, date, subject = (
            git("log", "-1", "--format=%h%x09%ad%x09%s", "--date=short", sha)
            .strip()
            .split("\t")
        )
        shortstat = git("show", "--shortstat", "--format=", sha).strip().splitlines()
        files = []
        for line in git("show", "--format=", "--name-status", sha).splitlines():
            if line.strip():
                parts = line.split("\t")
                files.append((parts[0], parts[1:]))
        commits.append(
            {
                "short": short,
                "date": date,
                "subject": subject,
                "body": git("log", "-1", "--format=%b", sha).strip(),
                "stat": shortstat[-1].strip() if shortstat else "",
                "files": files,
            }
        )
    return commits


def assign_eras(commits: list[dict]) -> None:
    starts = {start: index for index, (start, _, _) in enumerate(ERAS)}
    current = 0
    for commit in commits:
        current = starts.get(commit["short"], current)
        commit["era"] = current


def render(commits: list[dict]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# READ_FILES — how the project got here")
    add("")
    add("Every commit in the repository, oldest first, with the files it touched. Read")
    add("it to see which files carry the most history and in what order they changed.")
    add("")
    add(f"**{len(commits)} commits**, {commits[0]['date']} to {commits[-1]['date']}.")
    add("")
    add(f"Current as of `{commits[-1]['short']}`. Generated by")
    add("`scripts/generate_read_files.py` — re-run it after committing rather than")
    add("editing this file by hand, and commit the refresh on its own.")
    add("")
    add("Status codes: `added` · `modified` · `deleted` · `renamed`.")
    add("")
    add("---")
    add("")
    add("## Contents")
    add("")
    for index, (_, title, _) in enumerate(ERAS):
        era = [c for c in commits if c["era"] == index]
        if not era:
            continue
        span = f"{era[0]['short']} … {era[-1]['short']}"
        add(f"- [{title}](#{anchor(title)}) — {len(era)} commits ({span})")
    add("- [Files by number of commits](#files-by-number-of-commits)")
    add("- [Suggested reading order](#suggested-reading-order-for-the-current-state)")
    add("")
    add("---")
    add("")

    for index, (_, title, blurb) in enumerate(ERAS):
        era = [c for c in commits if c["era"] == index]
        if not era:
            continue
        add(f"## {title}")
        add("")
        add(blurb)
        add("")
        for commit in era:
            add(f"### `{commit['short']}` — {commit['subject']}")
            add("")
            add(f"*{commit['date']}* · {commit['stat']}")
            add("")
            if commit["body"]:
                summary = commit["body"].split("\n\n")[0].replace("\n", " ").strip()
                if summary:
                    add(f"> {summary}")
                    add("")
            for code, paths in commit["files"]:
                label = status_label(code)
                if label == "renamed" and len(paths) == 2:
                    add(f"- `{paths[0]}` → `{paths[1]}` *(renamed)*")
                else:
                    add(f"- `{paths[-1]}` *({label})*")
            add("")
        add("---")
        add("")

    touches: collections.Counter[str] = collections.Counter()
    for commit in commits:
        for _, paths in commit["files"]:
            touches[paths[-1]] += 1

    add("## Files by number of commits")
    add("")
    add("The files that changed most often are the ones whose design moved most.")
    add("Reading one top to bottom — `git log -p -- <path>` — is the fastest way")
    add("to see why it looks the way it does.")
    add("")
    add("| Commits | File |")
    add("|---:|---|")
    for path, count in sorted(touches.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < FREQUENCY_FLOOR:
            continue
        gone = "" if (REPO / path).exists() else " *(gone)*"
        add(f"| {count} | `{path}`{gone} |")
    add("")
    add("---")
    add("")
    add("## Suggested reading order for the current state")
    add("")
    add("Not chronological — this is the order that makes the code make sense, and it")
    add("mirrors the table at the end of [`docs/DESIGN.md`](docs/DESIGN.md).")
    add("")
    for position, (path, why) in enumerate(READING_ORDER, start=1):
        add(f"{position}. **`{path}`** — {why}")
    add("")
    return "\n".join(lines)


def main() -> int:
    commits = collect()
    assign_eras(commits)
    OUTPUT.write_text(render(commits))
    print(f"wrote {OUTPUT.relative_to(REPO)} — {len(commits)} commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
