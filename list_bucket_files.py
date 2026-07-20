"""
List the contents of the public gnomad-nc-constraint-v31-paper GCS bucket via
the JSON API (no gsutil, no auth needed -- the bucket is world-readable).

By default lists only the top level (like `ls`, using GCS's `delimiter`
support to stop at "directory" boundaries) since some subdirectories are huge
(e.g. context_prepared.ht/ alone has 38,000+ objects). Pass -recursive to
descend into every subdirectory instead, and -prefix to scope the listing to
one subdirectory (recursive listing without -prefix will enumerate the whole
bucket, which is slow and produces a very long listing).

Examples:
    python list_bucket_files.py                              # top-level only
    python list_bucket_files.py -prefix fig_tables/           # one dir, top-level
    python list_bucket_files.py -prefix fig_tables/ -recursive
    python list_bucket_files.py -recursive                    # whole bucket (slow)
"""
import argparse
import json
import urllib.request

BUCKET = "gnomad-nc-constraint-v31-paper"
API_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


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
            print(f"{name}  (directory)")
        else:
            print(f"{name}\t{human_size(size)}")
            n_files += 1
            total_size += size

    print(f"\n{n_files} files, {human_size(total_size)} total"
          + ("" if args.recursive else " (top-level listing only; pass -recursive to descend)"))


if __name__ == "__main__":
    main()
