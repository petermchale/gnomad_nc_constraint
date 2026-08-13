"""
Refresh fig5.ai's linked panel PDFs and save it, by driving Illustrator.

The figure is assembled in Illustrator from the per-panel PDFs the notebook writes, as
LINKS rather than embedded art. Rebuilding the figure therefore leaves the .ai stale:
Illustrator has to reload each link and the document has to be saved before the repo --
which is the source of truth for fig5.ai -- reflects the new panels. Doing that by hand
after every rebuild is the thing this script removes.

    .venv/bin/python fig5/resave_ai.py -dry_run   # what is stale, touching nothing
    .venv/bin/python fig5/resave_ai.py            # relink the stale ones and save

It talks to Illustrator over `osascript -e 'tell application ... to do javascript'`,
i.e. ExtendScript, which is the only way in: an .ai file stores a path and a cached
preview per link, and regenerating that preview outside the app is not something we can
do. So this needs Illustrator installed, and the first run raises a macOS Automation
permission prompt that has to be approved once.

TWO THINGS TO KNOW BEFORE TRUSTING IT.

  * Relinking preserves each placed item's frame, not the artwork's aspect ratio. The
    panels are saved with bbox_inches="tight", so a panel whose labels changed can come
    back with a slightly different bounding box and get stretched into the old frame.
    Eyeball the result the first time a panel's layout changes; the script prints every
    link it touched so there is a list to check.
  * It saves whatever state the document is in, including edits in progress. That is
    deliberate -- a document whose links just auto-refreshed is "dirty" in exactly the
    same way as one being edited, and the two cannot be told apart. fig5.ai is tracked,
    so `git checkout fig5/fig5.ai` undoes anything unwanted.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AI_PATH = os.path.join(HERE, "fig5.ai")
OUTPUT_DIR = os.path.join(HERE, "output")

# ExtendScript, run inside Illustrator. It reports back what it did -- see the string it
# builds at the bottom -- so the Python side prints facts rather than assumptions. The two
# %-slots are the .ai's path (as a JS string literal) and its mtime in epoch seconds.
#
# The document is found among the open ones before falling back to opening it: the usual
# case is that it is already open on screen, and opening a second copy of an open file is
# an error in Illustrator. A document this script opened itself is closed again; one that
# was already open is left exactly as it was, minus the save.
JSX = r"""
var target = %s;
var stale_after = %f;
var result = {opened: false, relinked: [], skipped: [], saved: false, note: ""};

app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

var doc = null;
for (var i = 0; i < app.documents.length; i++) {
    if (app.documents[i].fullName.fsName == target) { doc = app.documents[i]; break; }
}
if (doc == null) {
    doc = app.open(new File(target));
    result.opened = true;
}

for (var j = 0; j < doc.placedItems.length; j++) {
    var item = doc.placedItems[j];
    var f = item.file;
    if (f == null || !f.exists) { result.skipped.push("missing link"); continue; }
    // File.modified is a Date; compare against the .ai's own mtime, passed in as epoch
    // seconds. A link older than the document cannot be the reason it is stale.
    if (f.modified.getTime() / 1000.0 <= stale_after) { result.skipped.push(f.name); continue; }
    item.relink(f);
    result.relinked.push(f.name);
}

if (result.relinked.length > 0 || !doc.saved) {
    doc.save();
    result.saved = true;
}
if (result.opened) { doc.close(SaveOptions.DONOTSAVECHANGES); }
// Illustrator's ExtendScript engine is ES3 and has no JSON object, so the result comes
// back as a delimited string that parse_result() below turns into a dict.
("opened=" + (result.opened ? 1 : 0)
 + ";saved=" + (result.saved ? 1 : 0)
 + ";relinked=" + result.relinked.join(",")
 + ";skipped=" + result.skipped.join(","));
"""


def run_jsx(script: str) -> dict:
    """Hand one ExtendScript string to Illustrator and parse what it returns."""
    proc = subprocess.run(
        ["osascript", "-e",
         'tell application "Adobe Illustrator" to do javascript ' + applescript_str(script)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"osascript failed ({proc.returncode}):\n{proc.stderr.strip()}\n\n"
            "If this is a permissions error, approve Terminal (or your IDE) under\n"
            "System Settings > Privacy & Security > Automation > Adobe Illustrator.")
    return parse_result(proc.stdout.strip())


def parse_result(out: str) -> dict:
    """`k=v;k=v` from the JSX above, with the two list-valued keys split on commas."""
    fields = dict(kv.split("=", 1) for kv in out.split(";") if "=" in kv)
    if "saved" not in fields:
        raise RuntimeError(f"Illustrator returned something unparseable:\n{out}")
    return {"opened": fields["opened"] == "1", "saved": fields["saved"] == "1",
            "relinked": [n for n in fields.get("relinked", "").split(",") if n],
            "skipped": [n for n in fields.get("skipped", "").split(",") if n]}


def applescript_str(s: str) -> str:
    """A Python string as an AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def stale_links(ai_mtime: float) -> list[str]:
    """Panel PDFs newer than the .ai -- what the relink is for, computed without the app."""
    pdfs = [os.path.join(OUTPUT_DIR, f) for f in sorted(os.listdir(OUTPUT_DIR))
            if f.endswith(".pdf")]
    return [os.path.basename(p) for p in pdfs if os.path.getmtime(p) > ai_mtime]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-dry_run", action="store_true",
                    help="report which panels are newer than the .ai; touch nothing")
    args = ap.parse_args()

    if not os.path.exists(AI_PATH):
        print(f"no such file: {AI_PATH}", file=sys.stderr)
        return 1

    ai_mtime = os.path.getmtime(AI_PATH)
    stale = stale_links(ai_mtime)
    if not stale:
        print(f"fig5.ai is newer than every panel PDF in {os.path.relpath(OUTPUT_DIR)}/ "
              "-- nothing to do")
        return 0

    print(f"{len(stale)} panel PDF(s) newer than fig5.ai: {', '.join(stale)}")
    if args.dry_run:
        print("dry run -- Illustrator not contacted")
        return 0

    result = run_jsx(JSX % (applescript_json(AI_PATH), ai_mtime))
    if result["opened"]:
        print("opened fig5.ai (it was not already open)")
    if result["relinked"]:
        print(f"relinked: {', '.join(result['relinked'])}")
    print("saved fig5.ai" if result["saved"] else "nothing to save")
    print("Check the relinked panels: a changed bounding box is stretched into the old "
          "frame.\nUndo with: git checkout fig5/fig5.ai")
    return 0


def applescript_json(s: str) -> str:
    """A path as a JS string literal, for interpolation into JSX."""
    return json.dumps(s)


if __name__ == "__main__":
    raise SystemExit(main())
