"""
Tee a script's run to a log committed beside it, so its output can be read without
re-running it.

    import runlog

    with runlog.tee(LOG_PATH, "python list_bucket_files.py -depth 2",
                    "the public bucket's top level, expanded one level"):
        main_work()

Three things the log does that a shell redirect does not:

  - **records what produced it** -- the command, the commit the script was at, and
    whether that script was modified at the time. If the recorded SHA is not an
    ancestor of what you are reading, the log predates the code and does not describe
    it. Dirtiness is measured over the calling script alone, not a directory, so a log
    can be committed on top of its own subject without the stamp going stale.
  - **stays machine-independent** -- absolute paths are rewritten relative to the log's
    own directory, so a run on another machine diffs only where the numbers differ.
  - **stays readable** -- ANSI colour is stripped on the way to the file but left on the
    terminal, so a script can colourize for a human without polluting the committed copy.

`preconditions/report.py` does all of this and more (PASS/FAIL claims, a rendered
STATUS.md) but deliberately not for these callers: registering a script there gives it a
row among the claims fig5 rests on, and neither of these scripts is one. It keeps its own
copy of the tee rather than importing this module, because it is the audit backbone of
the repo and reaching up out of `preconditions/` to import from the root would make it
depend on a file nothing else in that directory needs.
"""
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _Tee:
    """Terminal gets the stream unchanged; the file gets it plain and path-scrubbed."""

    def __init__(self, stream, fh, base_dir: str):
        self.stream, self.fh, self.base_dir = stream, fh, base_dir

    def write(self, s: str) -> int:
        self.fh.write(_ANSI.sub("", s)
                      .replace(self.base_dir, ".")
                      .replace(os.path.expanduser("~"), "~"))
        return self.stream.write(s)

    def flush(self) -> None:
        self.fh.flush()
        self.stream.flush()

    def __getattr__(self, name):  # isatty, encoding, ... asked for by other libraries
        return getattr(self.stream, name)


def git_stamp(path: str) -> str:
    """The commit `path` ran at, marked `+dirty` if `path` itself was uncommitted."""
    repo = os.path.dirname(os.path.abspath(path))

    def git(*args) -> str:
        return subprocess.run(["git", "-C", repo, *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        dirty = bool(git("status", "--porcelain", "--", os.path.abspath(path)))
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@contextmanager
def tee(log_path: str, command: str, what: str, script: str | None = None):
    """
    Tee everything printed inside the block to `log_path`, under a three-line header.

    `command` is the invocation to record -- write out the flags actually used, since a
    log whose header does not match the run that produced it is worse than none.
    `script` defaults to the file `log_path` sits beside, which is the stamped subject.
    """
    script = script or os.path.splitext(log_path)[0] + ".py"
    base_dir = os.path.dirname(os.path.abspath(log_path))
    with open(log_path, "w") as fh:
        saved_stdout, sys.stdout = sys.stdout, _Tee(sys.stdout, fh, base_dir)
        try:
            print(f"$ {command}")
            print(f"# {what}")
            print(f"# ran at {git_stamp(script)}, "
                  f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
            yield
        finally:
            sys.stdout = saved_stdout
    print(f"wrote {os.path.basename(log_path)}")
