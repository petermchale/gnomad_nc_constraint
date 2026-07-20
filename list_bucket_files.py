"""
List the contents of the public gnomad-nc-constraint-v31-paper GCS bucket via
the JSON API (no gsutil, no auth needed -- the bucket is world-readable).

By default lists only the top level (like `ls`, using GCS's `delimiter`
support to stop at "directory" boundaries) since some subdirectories are huge
(e.g. context_prepared.ht/ alone has 38,000+ objects). Pass -recursive to
descend into every subdirectory instead, and -prefix to scope the listing to
one subdirectory (recursive listing without -prefix will enumerate the whole
bucket, which is slow and produces a very long listing).

Each directory entry also reports its total recursive size (not just its
immediate children's), which needs a full recursive listing of that
directory regardless of whether -recursive was passed for the outer listing
-- for huge directories like context_prepared.ht/ (~578 GB, 38,000+ objects)
this alone takes 100+ paginated API calls, so a plain top-level listing of
the bucket root is noticeably slower than it looks.

Examples:
    python list_bucket_files.py                              # top-level only
    python list_bucket_files.py -prefix fig_tables/           # one dir, top-level
    python list_bucket_files.py -prefix fig_tables/ -recursive
    python list_bucket_files.py -recursive                    # whole bucket (slow)
"""
import argparse
import json
import os
import sys
import urllib.request

BUCKET = "gnomad-nc-constraint-v31-paper"
API_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"

# Colorize only when writing to a real terminal (not a pipe/file) and NO_COLOR
# isn't set (https://no-color.org) -- otherwise ANSI codes would just pollute
# redirected output.
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str) -> str:
    return code if _USE_COLOR else ""


RESET = _c("\033[0m")
DIR = _c("\033[1;34m")     # bold blue, like `ls` directory coloring
COUNT = _c("\033[90m")     # gray
SIZE = _c("\033[32m")      # green
SUMMARY = _c("\033[1;36m")  # bold cyan


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def list_objects(prefix: str, recursive: bool):
    """Yields (name, size) tuples. Non-recursive listing also yields
    (dirname, None) tuples for subdirectory prefixes."""
    token = None
    while True:
        params = {"maxResults": "1000"}
        if prefix:
            params["prefix"] = prefix
        if not recursive:
            params["delimiter"] = "/"
        if token:
            params["pageToken"] = token
        url = API_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)

        for item in data.get("items", []):
            if item["name"] == prefix:
                continue
            yield item["name"], int(item.get("size", 0))
        for d in data.get("prefixes", []):
            yield d, None

        token = data.get("nextPageToken")
        if not token:
            break


def count_immediate_children(prefix: str):
    """Returns (n_files, n_subdirs) of the immediate (non-recursive) children
    of prefix, via the same delimiter-based listing as list_objects."""
    n_files = 0
    n_subdirs = 0
    for _, size in list_objects(prefix, recursive=False):
        if size is None:
            n_subdirs += 1
        else:
            n_files += 1
    return n_files, n_subdirs


def dir_total_size(prefix: str) -> int:
    """Returns the total size (bytes) of every file recursively under prefix
    -- not just its immediate children. Requires a full recursive listing, so
    this is slow for huge directories (e.g. context_prepared.ht/, ~578 GB
    across 38,000+ objects, needs ~100+ paginated API calls on its own)."""
    return sum(size for _, size in list_objects(prefix, recursive=True) if size is not None)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-prefix", default="",
        help="only list objects under this prefix, e.g. 'fig_tables/' (default: bucket root)")
    parser.add_argument(
        "-recursive", action="store_true",
        help="descend into subdirectories instead of stopping at the first '/' (default: off)")
    args = parser.parse_args()

    n_files = 0
    total_size = 0
    for name, size in list_objects(args.prefix, args.recursive):
        if size is None:
            n_child_files, n_child_subdirs = count_immediate_children(name)
            child_total_size = dir_total_size(name)
            print(f"{DIR}{name}{RESET}  {COUNT}(directory: {n_child_files} files, "
                  f"{n_child_subdirs} subdirs, {RESET}{SIZE}{human_size(child_total_size)}"
                  f"{COUNT} total){RESET}")
        else:
            print(f"{name}\t{SIZE}{human_size(size)}{RESET}")
            n_files += 1
            total_size += size

    tail = "" if args.recursive else " (top-level listing only; pass -recursive to descend)"
    print(f"\n{SUMMARY}{n_files} files, {human_size(total_size)} total{RESET}{tail}")


if __name__ == "__main__":
    main()
