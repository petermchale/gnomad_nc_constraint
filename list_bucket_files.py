"""
List the contents of the public gnomad-nc-constraint-v31-paper GCS bucket via
the JSON API (no gsutil, no auth needed -- the bucket is world-readable).

By default lists only the top level (like `ls`, using GCS's `delimiter`
support to stop at "directory" boundaries) since some subdirectories are huge
(e.g. context_prepared.ht/ alone has 38,000+ objects). Pass -recursive to
descend into every subdirectory instead, and -prefix to scope the listing to
one subdirectory (recursive listing without -prefix will enumerate the whole
bucket, which is slow and produces a very long listing).

Pass -depth N to also expand each directory's own contents inline (indented),
N levels deep -- e.g. -depth 2 shows every top-level directory's immediate
files/subdirs too, not just its file/subdir counts. Default is 1 (today's
behavior: counts only, no expansion). This is a bounded, indented expansion,
distinct from -recursive (which flattens the whole tree with no delimiter,
unbounded). Most directories in this bucket have only a handful of immediate
children, but a few internal Hail-table directories (some `*.ht/index/`
subdirs have 100,000+ entries) don't -- expansion refuses to descend into any
directory with more than MAX_EXPAND_CHILDREN immediate children (prints a
one-line notice instead), so -depth can't runaway into tens of thousands of
per-child API calls the way an early, uncapped version of this script did.

Pass -size to also report each directory's total recursive size (not just
its immediate children's). Off by default because it needs a full recursive
listing of that directory regardless of whether -recursive was passed for
the outer listing -- for huge directories like context_prepared.ht/
(~578 GB, 38,000+ objects) this alone takes 100+ paginated API calls, so
-size can make even a plain top-level listing very slow.

Examples:
    python list_bucket_files.py                              # top-level only
    python list_bucket_files.py -prefix fig_tables/           # one dir, top-level
    python list_bucket_files.py -prefix fig_tables/ -recursive
    python list_bucket_files.py -recursive                    # whole bucket (slow)
    python list_bucket_files.py -size                         # top-level + dir sizes (slow)
    python list_bucket_files.py -depth 2                      # top-level dirs, expanded one level
"""
import argparse
import json
import os
import sys
import urllib.request

BUCKET = "gnomad-nc-constraint-v31-paper"
API_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"

# -depth refuses to expand into (or count the children of) any directory with
# more than this many immediate children -- some Hail-internal directories
# (e.g. a *.ht/index/ subdir) have 100,000+, and counting each child
# individually is what made an earlier version of this script hang for
# minutes on a single directory.
MAX_EXPAND_CHILDREN = 30

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


def print_entry(name: str, size, indent: int, show_size: bool):
    """Prints one file or directory line at the given indent level. For a
    directory, also returns its (n_child_files, n_child_subdirs) counts."""
    pad = "  " * indent
    if size is None:
        n_child_files, n_child_subdirs = count_immediate_children(name)
        size_note = ""
        if show_size:
            child_total_size = dir_total_size(name)
            size_note = f", {RESET}{SIZE}{human_size(child_total_size)}{COUNT} total"
        print(f"{pad}{DIR}{name}{RESET}  {COUNT}(directory: {n_child_files} files, "
              f"{n_child_subdirs} subdirs{size_note}){RESET}")
        return n_child_files, n_child_subdirs
    else:
        print(f"{pad}{name}\t{SIZE}{human_size(size)}{RESET}")
        return None


def list_tree(prefix: str, depth: int, show_size: bool, indent: int = 0):
    """Prints prefix's immediate children; for each subdirectory found,
    recurses one level less (indenting), down to `depth` total levels.
    Refuses to expand (or count children of) any directory with more than
    MAX_EXPAND_CHILDREN immediate children -- prints a one-line notice
    instead, so a single huge Hail-internal directory can't make this hang."""
    children = []
    truncated = False
    for entry in list_objects(prefix, recursive=False):
        children.append(entry)
        if len(children) > MAX_EXPAND_CHILDREN:
            truncated = True
            break

    pad = "  " * indent
    if truncated:
        print(f"{pad}{COUNT}... more than {MAX_EXPAND_CHILDREN} entries under {prefix} -- "
              f"not expanding (use -prefix '{prefix}' to view directly){RESET}")
        return

    for name, size in children:
        print_entry(name, size, indent, show_size)
        if size is None and depth > 1:
            list_tree(name, depth - 1, show_size, indent + 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-prefix", default="",
        help="only list objects under this prefix, e.g. 'fig_tables/' (default: bucket root)")
    parser.add_argument(
        "-recursive", action="store_true",
        help="descend into subdirectories instead of stopping at the first '/' (default: off)")
    parser.add_argument(
        "-depth", type=int, default=1,
        help="expand this many levels deep, indenting subdirectory contents inline "
             "(default: 1, i.e. no expansion -- just top-level counts). Ignored if "
             "-recursive is passed.")
    parser.add_argument(
        "-size", action="store_true",
        help="also report each directory's total recursive size -- slow, needs a full "
             "recursive listing per directory (default: off)")
    args = parser.parse_args()

    n_files = 0
    total_size = 0
    if args.recursive:
        for name, size in list_objects(args.prefix, recursive=True):
            print_entry(name, size, indent=0, show_size=args.size)
            n_files += 1
            total_size += size
        tail = ""
    else:
        for name, size in list_objects(args.prefix, recursive=False):
            counts = print_entry(name, size, indent=0, show_size=args.size)
            if counts is None:
                n_files += 1
                total_size += size
            elif args.depth > 1:
                list_tree(name, args.depth - 1, args.size, indent=1)
        tail = " (top-level listing only; pass -recursive to descend, or -depth N to expand inline)"

    print(f"\n{SUMMARY}{n_files} files, {human_size(total_size)} total{RESET}{tail}")


if __name__ == "__main__":
    main()
