"""
Refresh fig5.ai's linked panel PDFs, save it, and re-export fig5.png -- by driving
Illustrator.

The figure is assembled in Illustrator from the per-panel PDFs the notebook writes, as
LINKS rather than embedded art. Rebuilding the figure therefore leaves the .ai stale:
Illustrator has to reload each link and the document has to be saved before the repo --
which is the source of truth for fig5.ai -- reflects the new panels. Then fig5.png, the
readable copy of the assembly, has to be exported from the saved document, since an .ai
renders as nothing on GitHub. Doing all of that by hand after every rebuild is the thing
this script removes.

    .venv/bin/python fig5/resave_ai.py -dry_run   # what is stale, touching nothing
    .venv/bin/python fig5/resave_ai.py            # relink, save, re-export the PNG
    .venv/bin/python fig5/resave_ai.py -no_png    # ... leaving fig5.png alone

fig5.ipynb's last cell calls refresh() itself, so a notebook run leaves the assembly
current without a second command. Running the script by hand is for the case where the
panels were rebuilt some other way, or where the first run's permission prompt needs a
terminal to appear in front of.

It talks to Illustrator over `osascript -e 'tell application ... to do javascript'`,
i.e. ExtendScript, which is the only way in: an .ai file stores a path and a cached
preview per link, and regenerating that preview outside the app is not something we can
do. So this needs Illustrator installed, and the first run raises a macOS Automation
permission prompt that has to be approved once.

THREE THINGS TO KNOW BEFORE TRUSTING IT.

  * Relinking preserves each placed item's frame, not the artwork's aspect ratio. The
    panels are saved with bbox_inches="tight", so a panel whose labels changed can come
    back with a slightly different bounding box and get stretched into the old frame.
    Eyeball the result the first time a panel's layout changes; the script prints every
    link it touched so there is a list to check.
  * It saves whatever state the document is in, including edits in progress. That is
    deliberate -- a document whose links just auto-refreshed is "dirty" in exactly the
    same way as one being edited, and the two cannot be told apart. Both outputs are
    tracked, so `git checkout fig5/fig5.ai fig5/fig5.png` undoes anything unwanted.
  * The PNG export settings are hardcoded (PNG_DPI below, transparent background,
    artboard-clipped) to reproduce the committed export rather than to redefine it. If
    you re-export by hand with different settings, change them here too or the next run
    will quietly put them back.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
AI_PATH = os.path.join(HERE, "fig5.ai")
PNG_PATH = os.path.join(HERE, "fig5.png")
OUTPUT_DIR = os.path.join(HERE, "output")

# The PNG beside the .ai is the readable copy of the assembled figure -- what a reader
# sees on GitHub, where an .ai renders as nothing. 300 dpi, artboard-clipped, transparent
# background: read back off the committed export (1438 x 1393 px for a 344.9 x 334.1 pt
# artboard is 300/72), so re-exporting reproduces it rather than redefining it.
PNG_DPI = 300

# ExtendScript, run inside Illustrator. It reports back what it did -- see the string it
# builds at the bottom -- so the Python side prints facts rather than assumptions. The
# %-slots are the .ai's path (as a JS string literal), its mtime in epoch seconds, the PNG
# path to export to ("" to skip), and the export scale as a percentage of 72 dpi.
#
# The document is found among the open ones before falling back to opening it: the usual
# case is that it is already open on screen, and opening a second copy of an open file is
# an error in Illustrator. A document this script opened itself is closed again; one that
# was already open is left exactly as it was, minus the save.
JSX = r"""
var target = %s;
var stale_after = %f;
var png_target = %s;
var png_scale = %f;
var result = {opened: false, relinked: [], skipped: [], links: [], saved: false,
              exported: false};

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
    result.links.push(f.name);
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

// PNG24 rather than PNG8: the figure is antialiased line art over transparency, which a
// 256-colour palette dithers. artBoardClipping crops to the artboard, so the export is
// the figure and not its bounding box plus whatever sits outside.
if (png_target != "") {
    var opts = new ExportOptionsPNG24();
    opts.antiAliasing = true;
    opts.transparency = true;
    opts.artBoardClipping = true;
    opts.horizontalScale = png_scale;
    opts.verticalScale = png_scale;
    doc.exportFile(new File(png_target), ExportType.PNG24, opts);
    result.exported = true;
}
if (result.opened) { doc.close(SaveOptions.DONOTSAVECHANGES); }
// Illustrator's ExtendScript engine is ES3 and has no JSON object, so the result comes
// back as a delimited string that parse_result() below turns into a dict.
("opened=" + (result.opened ? 1 : 0)
 + ";saved=" + (result.saved ? 1 : 0)
 + ";exported=" + (result.exported ? 1 : 0)
 + ";relinked=" + result.relinked.join(",")
 + ";skipped=" + result.skipped.join(",")
 + ";links=" + result.links.join(","));
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
    """`k=v;k=v` from the JSX above, with the list-valued keys split on commas."""
    fields = dict(kv.split("=", 1) for kv in out.split(";") if "=" in kv)
    if "saved" not in fields:
        raise RuntimeError(f"Illustrator returned something unparseable:\n{out}")
    return {"opened": fields["opened"] == "1", "saved": fields["saved"] == "1",
            "exported": fields.get("exported") == "1",
            "relinked": [n for n in fields.get("relinked", "").split(",") if n],
            "skipped": [n for n in fields.get("skipped", "").split(",") if n],
            "links": [n for n in fields.get("links", "").split(",") if n]}


def applescript_str(s: str) -> str:
    """A Python string as an AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def stamp_png_dpi(path: str, dpi: int = PNG_DPI) -> bool:
    """
    Write `dpi` into the PNG's pHYs chunk, and report whether it went in.

    Illustrator's *scripted* export writes no pHYs at all, where its export dialog writes
    one -- so without this the file is the same pixels but declares no resolution, and
    anything that places it by physical size (Word, Docs, LaTeX) falls back to 72 dpi and
    lays it out 4x too large. The scale is already applied to the pixels; this is metadata
    only.

    Done by rewriting the chunk list rather than through an imaging library, so the pixel
    data is copied byte for byte and never re-encoded. pHYs is 9 bytes -- x and y pixels
    per unit, then the unit, 1 meaning metres -- and must precede IDAT.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return False

    body = b"pHYs" + struct.pack(">IIB", round(dpi / 0.0254), round(dpi / 0.0254), 1)
    chunk = (struct.pack(">I", len(body) - 4) + body
             + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    out, pos, inserted = bytearray(data[:8]), 8, False
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype, end = data[pos + 4:pos + 8], pos + 12 + length
        if ctype == b"pHYs":          # drop any existing one rather than duplicating it
            pos = end
            continue
        if ctype == b"IDAT" and not inserted:
            out += chunk
            inserted = True
        out += data[pos:end]
        pos = end
    if not inserted:
        return False
    with open(path, "wb") as fh:
        fh.write(out)
    return True


def stale_links(ai_mtime: float) -> list[str]:
    """Panel PDFs newer than the .ai -- what the relink is for, computed without the app."""
    pdfs = [os.path.join(OUTPUT_DIR, f) for f in sorted(os.listdir(OUTPUT_DIR))
            if f.endswith(".pdf")]
    return [os.path.basename(p) for p in pdfs if os.path.getmtime(p) > ai_mtime]


def refresh(dry_run: bool = False, quiet_if_absent: bool = False,
            png: bool = True) -> int:
    """
    The whole job, as one call: relink fig5.ai's stale panels, save it, and re-export
    fig5.png from it. Separate from main() so fig5.ipynb's last cell can invoke it
    directly.

    Two independent staleness questions, because the second outlives the first: a panel
    newer than the .ai needs relinking, and a .ai newer than the PNG needs exporting.
    Relinking implies the second (the save moves the .ai's mtime), but not the reverse --
    a save made by hand in Illustrator leaves the PNG behind with nothing to relink.

    `quiet_if_absent` is for the notebook: it must stay runnable by someone with no
    Illustrator and no .ai, so there the absence of either is a printed notice rather than
    a failure. From the command line it is an error worth seeing.
    """
    if not os.path.exists(AI_PATH):
        msg = f"no such file: {AI_PATH}"
        if quiet_if_absent:
            print(msg + " -- nothing to refresh")
            return 0
        print(msg, file=sys.stderr)
        return 1

    ai_mtime = os.path.getmtime(AI_PATH)
    stale = stale_links(ai_mtime)
    png_stale = png and (not os.path.exists(PNG_PATH)
                         or os.path.getmtime(PNG_PATH) < ai_mtime)
    if stale:
        print(f"{len(stale)} panel PDF(s) newer than fig5.ai: {', '.join(stale)}")
    if png_stale and not stale:
        print("fig5.ai is newer than fig5.png -- re-exporting")
    if not stale and not png_stale:
        print(f"fig5.ai is newer than every panel PDF in {os.path.relpath(OUTPUT_DIR)}/, "
              "and fig5.png is newer than fig5.ai -- nothing to do")
        return 0
    if dry_run:
        print("dry run -- Illustrator not contacted")
        return 0

    # Exported whenever the run does anything at all: a relink is always followed by a
    # save, which by itself leaves the PNG stale.
    try:
        result = run_jsx(JSX % (applescript_json(AI_PATH), ai_mtime,
                               applescript_json(PNG_PATH) if png else '""',
                               100.0 * PNG_DPI / 72.0))
    except RuntimeError as e:
        if quiet_if_absent:
            print(f"could not reach Illustrator, so fig5.ai is still stale:\n{e}")
            return 0
        raise

    if result["opened"]:
        print("opened fig5.ai (it was not already open)")
    if result["relinked"]:
        print(f"relinked: {', '.join(result['relinked'])}")
    absent = [f for f in stale if f not in result["links"]]
    if absent:
        print(f"not linked in fig5.ai, so left alone: {', '.join(absent)}")
    print("saved fig5.ai" if result["saved"] else "nothing to save")
    if result["exported"]:
        stamped = stamp_png_dpi(PNG_PATH)
        print("exported fig5.png" + (f" at {PNG_DPI} dpi" if stamped
                                     else f" (could not stamp {PNG_DPI} dpi into it)"))
    if result["relinked"]:
        print("Check the relinked panels: a changed bounding box is stretched into the "
              "old frame.")
    print("Undo with: git checkout fig5/fig5.ai fig5/fig5.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-dry_run", action="store_true",
                    help="report what is stale; touch nothing")
    ap.add_argument("-no_png", action="store_true",
                    help="relink and save only, leaving fig5.png alone")
    args = ap.parse_args()
    return refresh(dry_run=args.dry_run, png=not args.no_png)


def applescript_json(s: str) -> str:
    """A path as a JS string literal, for interpolation into JSX."""
    return json.dumps(s)


if __name__ == "__main__":
    raise SystemExit(main())
