"""
Make a precondition check's outcome readable without running it.

Every script here downloads multi-GB artifacts and prints a wall of numbers. A reader
of the repo -- a reviewer, a co-author, us in six months -- should not have to reproduce
that to learn whether the checks passed. So each script wraps its body in a `Report`,
which does three things:

  1. tees everything it prints to `output/<check>.log`, committed, so the full transcript
     of a real run is in the repo;
  2. collects explicit PASS/FAIL claims -- a check's conclusion is a list of booleans,
     not a paragraph a reader has to interpret;
  3. rewrites `output/STATUS.md` (rendered from `output/status.json`) with one row per
     check and every claim spelled out.

Read `output/STATUS.md` first. It is generated, never edited by hand.

Two honesty properties matter more than the formatting:

  - **A check that has never run says so.** The table is rendered from CHECKS below, not
    from whatever happens to be in status.json, so a check that was added and never run
    appears as "not run" rather than silently vanishing.
  - **A stale log is detectable.** Each row records the commit the check ran against and
    whether the checked source was dirty at the time. If that SHA is not an ancestor of
    what you are reading, the log predates the code and proves nothing about it.

Absolute paths are rewritten to repo-relative in the log, so a run on another machine
produces a diff only where the numbers actually differ.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(HERE, "output")
STATUS_JSON = os.path.join(OUTPUT_DIR, "status.json")
STATUS_MD = os.path.join(OUTPUT_DIR, "STATUS.md")

# The complete set of checks, so STATUS.md can distinguish "failed" from "never run".
# Keys are Report names; a script with independently-runnable checks registers one key
# per check (validate.py does, since its two halves cost seconds and minutes).
CHECKS = {
    "verify_expected_r1": (
        "the published expected-count table really is the context-only, r = 1 one, and "
        "describes the same windows as the published constraint table",
        ".venv/bin/python preconditions/verify_expected_r1.py",
    ),
    "verify_logit_predict_behavior": (
        "the operative adjustment is a ratio of probabilities, not the logit ratio the "
        "Methods state",
        ".venv/bin/python preconditions/verify_logit_predict_behavior.py",
    ),
    "verify_missing_utils_files": (
        "the multivariate PCA+logit fit behind r(w) is genuinely unpublished, and the "
        "three utility modules are not missing",
        ".venv/bin/python preconditions/verify_missing_utils_files.py",
    ),
    "verify_training_set_counts": (
        "the four shipped training tables are the training set the paper describes",
        ".venv/bin/python preconditions/verify_training_set_counts.py",
    ),
    "validate.coefficients": (
        "our univariate stage reproduces Chen et al.'s fitted coefficients and their "
        "selected-feature set",
        ".venv/bin/python preconditions/validate.py -check coefficients",
    ),
    "validate.expected": (
        "our full-population refit reproduces the published genome-wide expected counts",
        ".venv/bin/python preconditions/validate.py -check expected",
    ),
}

# Prose belonging to a SET of checks rather than to any one of them, so it has no natural
# home in CHECKS above or in a claim. Rendered after the table. Keep these to things a
# reader of STATUS.md would otherwise get wrong -- two rows that look like the same check,
# a claim that looks like independent corroboration of another -- and put the full argument
# in preconditions/README.md rather than restating it here.
NOTES = [(
    "`verify_expected_r1` and `validate.expected` are not the same check",
    """Both end in a genome-wide diff of expected counts, and neither computes E1 --
Chen et al.'s E1 is an input to both. `verify_expected_r1` compares E1 against E1, both
theirs, by re-aggregating the per-context export; what is under test is the E1 table's
IDENTITY as the pre-adjustment one, and no model runs. `validate.expected` compares E2
against E2, ours against theirs, after the full refit; what is under test is r, with E1
taken as given.

So the second cannot stand in for the first: E1 is a common factor on both of its sides
and cancels. Were the E1 table secretly post-adjustment, `validate.expected` would still
pass at Pearson r = 1.000000, both sides carrying the identical contamination, while
fig5 panel A's "before" curve was silently wrong. Full contrast in `../README.md`.""",
)]

_PASS, _FAIL, _ERROR, _NOT_RUN = "PASS", "FAIL", "ERROR", "not run"
_MARK = {_PASS: "PASS", _FAIL: "**FAIL**", _ERROR: "**ERROR**", _NOT_RUN: "_not run_"}


def _git_state() -> dict:
    """
    The commit the check ran against, plus whether the code it exercises was modified.

    Dirtiness is measured over `preconditions/` and `gnocchi_bias/` only, excluding
    `preconditions/output/` -- the logs themselves are always about to change, and an
    unrelated edit elsewhere in the repo says nothing about whether this log is current.
    """
    def git(*args) -> str:
        return subprocess.run(["git", "-C", REPO_ROOT, *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {
            "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "preconditions",
                              "gnocchi_bias", ":(exclude)preconditions/output")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": False}


class _Tee:
    """Writes to the terminal unchanged and to the log with machine-specific paths removed."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s: str) -> int:
        self.fh.write(s.replace(REPO_ROOT, ".").replace(os.path.expanduser("~"), "~"))
        return self.stream.write(s)

    def flush(self) -> None:
        self.fh.flush()
        self.stream.flush()

    def __getattr__(self, name):  # isatty, encoding, ... asked for by other libraries
        return getattr(self.stream, name)


class Report:
    """
    Context manager: tee this check's output to a committed log and record its claims.

        with Report("verify_x") as rep:
            rep.claim(max_diff < 1e-9, f"regenerated table matches ({max_diff:.1e})")

    Claim text should carry the measured number, not just the verdict -- STATUS.md is
    read by someone deciding whether to trust the result, and "PASS" alone is not
    evidence. State the threshold too wherever one was chosen rather than derived.

    Exits non-zero if any claim failed, after writing the log and STATUS.md: a failing
    precondition must be recorded, not swallowed, and must not be mistaken for a pass by
    a script that runs these in sequence.
    """

    def __init__(self, name: str, exit_on_failure: bool = True):
        if name not in CHECKS:
            raise KeyError(f"{name!r} is not in report.CHECKS; add it so STATUS.md can "
                           "show it as 'not run' before its first run")
        self.name = name
        self.exit_on_failure = exit_on_failure
        self.claims: list[tuple[bool, str]] = []
        self._t0 = 0.0

    def claim(self, ok, text: str) -> bool:
        ok = bool(ok)
        self.claims.append((ok, text))
        return ok

    def __enter__(self) -> "Report":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._t0 = datetime.now(timezone.utc).timestamp()
        self._fh = open(os.path.join(OUTPUT_DIR, f"{self.name}.log"), "w")
        self._saved_stdout = sys.stdout
        sys.stdout = _Tee(self._saved_stdout, self._fh)
        print(f"$ {CHECKS[self.name][1]}")
        print(f"# {CHECKS[self.name][0]}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = datetime.now(timezone.utc).timestamp() - self._t0
        if exc_type is not None:
            verdict = _ERROR
            print(f"\n{exc_type.__name__}: {exc}")
        elif not self.claims:
            # A check whose body ran to completion without asserting anything is a bug in
            # the check, and must not read as a pass.
            verdict = _ERROR
            print("\nno claims recorded -- a check that asserts nothing cannot pass")
        else:
            verdict = _PASS if all(ok for ok, _ in self.claims) else _FAIL

        print(f"\n{verdict}: {sum(ok for ok, _ in self.claims)}/{len(self.claims)} claims, "
              f"{elapsed:.0f}s")
        for ok, text in self.claims:
            print(f"  [{'PASS' if ok else 'FAIL'}] {text}")

        sys.stdout = self._saved_stdout
        self._fh.close()
        _update_status({
            "name": self.name,
            "verdict": verdict,
            "claims": [{"ok": ok, "text": text} for ok, text in self.claims],
            "seconds": round(elapsed, 1),
            "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            **_git_state(),
        })
        print(f"wrote preconditions/output/{self.name}.log and output/STATUS.md")

        # Only convert a clean run into a non-zero exit. If the body raised, let that
        # exception propagate with its own traceback rather than replacing it with a bare
        # SystemExit -- the record is already written either way.
        if exc_type is None and verdict != _PASS and self.exit_on_failure:
            raise SystemExit(1)
        return False


def _update_status(entry: dict) -> None:
    state = {}
    if os.path.exists(STATUS_JSON):
        with open(STATUS_JSON) as f:
            state = json.load(f)
    state[entry["name"]] = entry
    with open(STATUS_JSON, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    _render(state)


def _render(state: dict) -> None:
    run = [state[n] for n in CHECKS if n in state]
    n_pass = sum(e["verdict"] == _PASS for e in run)
    if n_pass == len(CHECKS):
        headline = f"**All {len(CHECKS)} precondition checks pass** " \
                   f"({sum(len(e['claims']) for e in run)} claims)."
    else:
        headline = (f"**{n_pass} of {len(CHECKS)} precondition checks pass** — "
                    f"{len(CHECKS) - len(run)} not run, "
                    f"{sum(e['verdict'] != _PASS for e in run)} failing.")

    lines = [
        "# Precondition status",
        "",
        headline,
        "",
        "Generated by `preconditions/report.py` — each check rewrites its own row when it",
        "runs, so rows can be of different ages. Do not edit by hand.",
        "",
        "The `code` column is the commit each check ran against. If it is not an ancestor",
        "of the commit you are reading, that log predates this code and does not vouch for",
        "it; `+dirty` means `preconditions/` or `gnocchi_bias/` had uncommitted edits at",
        "run time. Full transcripts are the `.log` files beside this one.",
        "",
        "| check | verdict | claims | last run | code | what it checks |",
        "|---|---|---|---|---|---|",
    ]
    for name, (what, _cmd) in CHECKS.items():
        e = state.get(name)
        if e is None:
            lines.append(f"| `{name}` | {_MARK[_NOT_RUN]} | — | — | — | {what} |")
            continue
        ok = sum(c["ok"] for c in e["claims"])
        code = e["commit"] + ("+dirty" if e.get("dirty") else "")
        lines.append(f"| [`{name}`]({name}.log) | {_MARK[e['verdict']]} | "
                     f"{ok}/{len(e['claims'])} | {e['when']} | `{code}` | {what} |")

    for heading, body in NOTES:
        lines += ["", f"## {heading}", "", body]

    lines += ["", "## Claims", ""]
    for name, (_what, cmd) in CHECKS.items():
        e = state.get(name)
        lines += [f"### `{name}`", "", f"```\n{cmd}\n```", ""]
        if e is None:
            lines += ["Not run.", ""]
            continue
        for c in e["claims"]:
            lines.append(f"- {'PASS' if c['ok'] else '**FAIL**'} — {c['text']}")
        lines += ["", f"_{e['verdict']}, {e['seconds']:.0f}s, {e['when']}._", ""]

    with open(STATUS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    # Re-render STATUS.md from status.json without re-running anything -- useful after
    # editing a check's description or adding a check to CHECKS.
    with open(STATUS_JSON) as f:
        _render(json.load(f))
    print(f"re-rendered {STATUS_MD}")
