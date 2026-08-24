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

It is meant to have nothing to do on most runs. Panels are written through save_panel()
below, which suppresses the PDF /CreationDate (matplotlib's only run-to-run
nondeterminism) and then writes only when the bytes differ, and staleness is judged
against fig5.ai.links.json, a record of what each panel hashed to when the .ai was last
saved. So a rebuild that changes no artwork touches no file and leaves the .ai alone --
and an asterisk on the .ai's tab in Illustrator means a panel really did change.

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
import hashlib
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
# background: read back off the committed export (1438 x 1392 px for a 344.9 x 334.1 pt
# artboard is 300/72), so re-exporting reproduces it rather than redefining it.
PNG_DPI = 300

# ExtendScript, run inside Illustrator. It reports back what it did -- see the string it
# builds at the bottom -- so the Python side prints facts rather than assumptions. The
# %-slots are the .ai's path (as a JS string literal), the semicolon-joined names of the
# links to relink, the PNG path to export to ("" to skip), and the export scale as a
# percentage of 72 dpi.
#
# WHICH LINKS TO RELINK IS DECIDED ON THE PYTHON SIDE and passed in by name. This used to
# be a second, independent mtime comparison here, which could disagree with the Python
# one -- and does, now that Python compares content: a panel restored from git can hold
# different artwork behind an older mtime, and this side would have skipped it.
#
# The document is found among the open ones before falling back to opening it: the usual
# case is that it is already open on screen, and opening a second copy of an open file is
# an error in Illustrator. A document this script opened itself is closed again; one that
# was already open is left exactly as it was, minus the save.
JSX = r"""
var target = %s;
var stale_names = ";" + %s + ";";
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
    if (stale_names.indexOf(";" + f.name + ";") < 0) { result.skipped.push(f.name); continue; }
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
    try:
        proc = subprocess.run(
            ["osascript", "-e",
             'tell application "Adobe Illustrator" to do javascript ' + applescript_str(script)],
            capture_output=True, text=True)
    except FileNotFoundError as e:
        # No osascript: not a Mac. Raised as RuntimeError, not left as FileNotFoundError,
        # because refresh(quiet_if_absent=True) catches RuntimeError to keep fig5.ipynb
        # runnable without Illustrator -- and on Linux that is the WHOLE reason the
        # notebook would otherwise die in its last cell, after the long run, on HPC.
        raise RuntimeError(
            "osascript not found, so Illustrator cannot be reached: this step is "
            "macOS-only. Copy fig5/output/*.pdf back to the Mac and assemble there "
            "(fig5/RUNBOOK.md step 8).") from e
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


# ---------------------------------------------------------- writing panels
#
# TWO RULES KEEP fig5.ai FROM GOING STALE FOR NO REASON, which is what this pair of
# functions is for. Before them, every rebuild rewrote every panel PDF with a fresh
# /CreationDate, so the .ai looked stale even when no artwork had changed -- and an .ai
# open in Illustrator would auto-relink and go dirty (the `*` in its tab) on a rebuild
# that changed nothing.
#
#   1. Suppress /CreationDate. It is the ONLY nondeterminism in matplotlib's PDF output
#      (measured: identical figures hash identically with it gone, and PNG output was
#      already deterministic), so two runs of an unchanged panel now produce identical
#      bytes.
#   2. Having made that true, write only when the bytes differ. An unchanged panel keeps
#      its old mtime, so nothing downstream -- the .ai, git -- sees a change that is not
#      one.
#
# The pair also removes the /CreationDate-only diffs that used to have to be checked out
# of a commit by hand, and with them the mtime pitfall that followed from doing so.

PDF_METADATA = {"CreationDate": None}


def write_if_changed(path: str, data: bytes) -> bool:
    """Write `data` to `path` only if it differs from what is there. True if written."""
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if fh.read() == data:
                return False
    with open(path, "wb") as fh:
        fh.write(data)
    return True


def save_panel(fig, stem: str, dpi: int = 200) -> list[str]:
    """
    Write `{stem}.pdf` (vector, for the Illustrator assembly) and `{stem}.png` (for
    reading), skipping either whose bytes are unchanged. Returns the paths written, which
    is what the notebook prints -- an empty list means the panel is already current, and
    is the normal result of re-running the notebook without changing anything.
    """
    import io

    written = []
    for ext, kw in ((".pdf", {"metadata": PDF_METADATA}), (".png", {"dpi": dpi})):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext[1:], bbox_inches="tight", **kw)
        if write_if_changed(f"{stem}{ext}", buf.getvalue()):
            written.append(f"{stem}{ext}")
    return written


# --------------------------------------------------------- staleness checks

# What each panel PDF hashed to when fig5.ai was last saved by this script. Tracked
# beside the .ai, because it describes the .ai: with it, "is the assembly current?" is
# answerable from content, by anyone, without opening Illustrator.
#
# Content and not mtime, because mtimes move for reasons that have nothing to do with the
# artwork -- `git checkout` of a panel, a stash pop, a rebase -- and each of those used to
# send the next run relinking a panel to content already in the document, dirtying the .ai
# again. Falls back to mtime when the manifest is missing (before the first save through
# this script) so the check still works, just less exactly.
LINKS_MANIFEST = os.path.join(HERE, "fig5.ai.links.json")


def panel_pdfs() -> list[str]:
    return [os.path.join(OUTPUT_DIR, f) for f in sorted(os.listdir(OUTPUT_DIR))
            if f.endswith(".pdf")]


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_links_manifest() -> dict:
    if not os.path.exists(LINKS_MANIFEST):
        return {}
    with open(LINKS_MANIFEST) as fh:
        return json.load(fh)


def write_links_manifest() -> None:
    """
    Record what the panels hash to now -- written whenever refresh() has reconciled the
    assembly, which includes the case where it found nothing to reconcile. A save made
    by hand in Illustrator lands there: the document already holds the current panels, so
    the next run should record that rather than stay on the mtime fallback forever.

    It covers every panel PDF, including supp_fig7.pdf, which is not a link in fig5.ai at
    all. Recording it is deliberate -- it is what stops a rebuilt supporting figure from
    being reported as stale on every run when there is nothing in the assembly to do
    about it.
    """
    manifest = {os.path.basename(p): _sha(p) for p in panel_pdfs()}
    with open(LINKS_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")


def stale_links(ai_mtime: float) -> list[str]:
    """
    Panel PDFs whose content is not what fig5.ai was last saved against -- what the
    relink is for, computed without the app. Falls back to "newer than the .ai" when no
    manifest has been written yet.
    """
    manifest = read_links_manifest()
    if not manifest:
        return [os.path.basename(p) for p in panel_pdfs()
                if os.path.getmtime(p) > ai_mtime]
    return [os.path.basename(p) for p in panel_pdfs()
            if manifest.get(os.path.basename(p)) != _sha(p)]


def refresh(dry_run: bool = False, quiet_if_absent: bool = False,
            png: bool = True) -> int:
    """
    The whole job, as one call: relink fig5.ai's stale panels, save it, and re-export
    fig5.png from it. Separate from main() so fig5.ipynb's last cell can invoke it
    directly.

    Two independent staleness questions, because the second outlives the first: a panel
    whose content the .ai does not hold needs relinking, and a .ai newer than the PNG
    needs exporting.
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
        how = "content differs from" if read_links_manifest() else "newer than"
        print(f"{len(stale)} panel PDF(s) {how} what fig5.ai was saved against: "
              f"{', '.join(stale)}")
    if png_stale and not stale:
        print("fig5.ai is newer than fig5.png -- re-exporting")
    if not stale and not png_stale:
        print(f"every panel PDF in {os.path.relpath(OUTPUT_DIR)}/ matches fig5.ai, "
              "and fig5.png is newer than fig5.ai -- nothing to do")
        write_links_manifest()
        return 0
    if dry_run:
        print("dry run -- Illustrator not contacted")
        return 0

    # Exported whenever the run does anything at all: a relink is always followed by a
    # save, which by itself leaves the PNG stale.
    try:
        result = run_jsx(JSX % (applescript_json(AI_PATH),
                               applescript_json(";".join(stale)),
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
    print("saved fig5.ai" if result["saved"] else "nothing to save (already saved)")
    # For every panel, not just the relinked ones: the unchanged ones' hashes are what
    # let the next run leave them alone. Written after the run rather than only after a
    # save, so a document saved by hand in Illustrator -- nothing here to save, links
    # already current -- still gets recorded.
    write_links_manifest()
    print(f"recorded {os.path.basename(LINKS_MANIFEST)}")
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
