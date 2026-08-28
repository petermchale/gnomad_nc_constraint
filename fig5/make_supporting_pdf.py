"""
The executed notebook, as a PDF fit to submit as supporting text.

WHY A SCRIPT AND NOT `nbconvert --to pdf`. A raw conversion of fig5.ipynb is a printout
of a working notebook: code cells, absolute HPC paths, polars table dumps, "wrote
fig5B.neutral.pdf", and two sections (Configuration, Refresh the Illustrator assembly)
that are about operating this repo rather than about the science. What a reader of the
supplement needs is the derivation, the figures, and the numbers the captions quote. This
script keeps exactly those, and it does it as a FILTER over the executed notebook rather
than as a second copy of its prose -- so fig5.ipynb stays the single source and this
never goes stale against it.

Three things it does that a flag cannot:

  * Panels come from `output/*.pdf`, not from the inline PNGs. The inline images are
    ~100 dpi rasters of the same figures; the panel PDFs are the vector files the
    manuscript itself is assembled from, so the supplement gets print-quality artwork
    for free. Matched by ORDER (the n-th figure-bearing cell to the n-th entry of
    PANELS), which is checked rather than assumed.
  * Each figure carries its manuscript caption, read from `captions.txt` -- so the
    supplement and the paper cannot disagree about what a panel shows.
  * Stream output is filtered, not dropped. The printed numbers ARE results (the
    window counts, the z sanity check, the caption numbers); the paths, the "wrote"
    lines and the polars tables are not.

Usage:

    .venv/bin/python fig5/make_supporting_pdf.py                # the run config.py points at
    .venv/bin/python fig5/make_supporting_pdf.py -keep_tex      # leave the .tex for hand-editing

Requires xelatex (MacTeX) and pandoc. The venv carries pandoc via `pypandoc_binary`,
symlinked to .venv/bin/pandoc; nbconvert finds it on PATH.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import nbformat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TITLE = ("Gnocchi's GC bias is introduced by its regional adjustment, "
         "and that adjustment is fit on the wrong population")
AUTHORS = "McHale, Goldberg \\& Quinlan"

# Sections that are about operating the repo, not about the result. Matched against the
# markdown cell's own "## " heading; the cells that follow one, up to the next heading,
# go with it.
DROP_SECTIONS = {
    "Configuration",                      # which two files to set, and where
    "Refresh the Illustrator assembly",   # relinking fig5.ai
}

# Headings whose wording is internal ("for the caption" is an instruction to us, not a
# statement to a reader). The section stays; only its title changes.
RETITLE = {
    "Numbers for the caption": "Values quoted in the figure captions",
    "Caveats that belong in the caption": "Caveats",
}

# Prose that only makes sense inside a notebook. Applied to markdown source; each must
# match exactly once across the notebook, so a rewrite that silently stops applying is
# an error rather than a no-op.
REWRITES = [
    ("Five panels, one argument. Each is written to `output/fig5{A..E}.pdf` as a standalone\n"
     "vector file for assembly in Illustrator.",
     "Five panels, one argument."),
    ("Everything quoted in the caption should come from here, not from memory.",
     "Every number the captions quote is computed here."),
    # Where this section's derivation came from is repo history, not a result.
    ("*Migrated from `chen_formula/chen_formula.tex` \u00a7\u00a71\u20135 (Peter McHale), which this notebook\nreplaces.*",
     ""),
    # The last caveat's tail is a cost estimate for rerunning this repo. The caveat
    # itself -- that the two window sets are different populations -- stays.
    ("\n  Switching recomputes everything: A, B and E directly, D through the `scored` refit\n"
     "  (`sizematched` is no longer drawn there, but panel E still reports it), and C through\n"
     "  both its bottom band and the shared GC bin edges,\n"
     "  which span the window set's own GC range. Two operational costs. The refits must be\n"
     "  rerun (`config.check()` refuses one stamped with a different value, naming the\n"
     "  command), and since they are keyed by population alone, one window set's refits\n"
     "  overwrite the other's \u2014 switching back means rerunning again, ~6 min each. The panel-C\n"
     "  and CpG caches in `output/` are keyed by a fingerprint of the edges and the window\n"
     "  set, so those two do coexist.",
     ""),
]

# Printed output survives only under this heading. Everywhere else it is a progress
# report -- row counts, file names, a polars frame echoed back -- which belongs in a
# terminal and not in a supplement. Under this one it is the result: the numbers the
# captions quote, computed rather than remembered.
KEEP_OUTPUT_SECTION = "Numbers for the caption"

# The n-th figure-bearing code cell gets the n-th of these: panel PDF stem, the
# captions.txt paragraph that belongs to it, and the name it is called by in the paper.
# The names are the manuscript's own, not a numbering of this document -- a reader must
# be able to carry "Fig. 5C" between the two -- so the captions here are UNNUMBERED.
PANELS = [
    ("fig5A", "A", "Fig. 5A"), ("fig5B", "B", "Fig. 5B"), ("fig5C", "C", "Fig. 5C"),
    ("fig5D", "D", "Fig. 5D"), ("fig5E", "E", "Fig. 5E"),
    ("supp_fig7", "S7", "Supporting Fig. 7"),
]

# Stream lines that report on the run rather than on the data.
NOISE = re.compile(
    r"^(repo root:|wrote |reusing |DEPLETION_RANK_BED|NEUTRAL_WINDOWS_BED|"
    r"\d+ panel PDF|could not reach Illustrator|osascript not found|"
    r"no such file)")
# A polars frame printed to stdout: `shape: (20, 7)` then a box-drawn table.
TABLE_START = re.compile(r"^shape: \(")
TABLE_CHARS = set("┌┬┐├┼┤└┴┘│─╞╪╡")


def _captions(path):
    """
    captions.txt as {key: paragraph}. It is one paragraph per line, blank-line separated:
    a title, a preamble, then (A)...(E); then Supporting Figure 7's own title, preamble,
    (A)-(D) and a closing note. The five main panels map to their letter; Supporting
    Figure 7 is one figure here, not four, so everything after its title is joined.
    """
    paras = [p.strip() for p in open(path).read().split("\n") if p.strip()]
    out, supp, in_supp = {}, [], False
    for p in paras:
        if p.startswith("Supporting Figure 7 |"):
            in_supp = True
            continue
        if in_supp:
            supp.append(p)
        else:
            m = re.match(r"\(([A-E])\)\s*", p)
            if m:
                # The panel letter is carried by the figure's own name here, so the
                # caption's "(A) " prefix would say it twice.
                out[m.group(1)] = p[m.end():]
    out["S7"] = " ".join(supp)
    return out


def _promote_headings(src):
    """
    `## X` -> `# X`, and so on down. The notebook's H1 becomes the LaTeX \\title, which
    would otherwise leave every section one level too deep -- a document whose top-level
    structure is \\subsection. Fenced blocks are skipped: a leading `#` inside one is a
    shell comment, not a heading.
    """
    out, fenced = [], False
    for line in src.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            line = re.sub(r"^(#{1,5})#(?= \S)", r"\1", line)
        out.append(line)
    return "".join(out)


def _escape(s):
    """Caption prose into LaTeX. captions.txt is deliberately plain ASCII already."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def _clean_stream(text):
    """Drop run-report lines and any printed dataframe; keep every printed number."""
    kept, skipping = [], False
    for line in text.splitlines():
        if TABLE_START.match(line.strip()):
            skipping = True
            continue
        if skipping:
            # A table ends at its bottom rule; anything else box-drawn is still table.
            if line.strip() and not (set(line.strip()) & TABLE_CHARS):
                skipping = False
            else:
                continue
        if NOISE.match(line.strip()):
            continue
        kept.append(line)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _figure(pdf_path, caption, name):
    """
    One float, captioned the way the paper captions it: `Fig. 5C | ...`, unnumbered.
    \\caption* rather than \\caption because a second numbering ("Figure 3:" over a panel
    the text calls Fig. 5C) is a numbering to get wrong.
    """
    return ("\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth,height=0.62\\textheight,keepaspectratio]{{{pdf_path}}}\n"
            f"\\caption*{{\\textbf{{{name}}} $|$ {caption}}}\n\\end{{figure}}")


def build(nb_path, out_dir, suffix, keep_tex):
    nb = nbformat.read(nb_path, as_version=4)
    captions = _captions(os.path.join(HERE, "captions.txt"))

    cells, drop_section, keep_output, n_fig = [], False, False, 0
    for cell in nb.cells:
        src = "".join(cell.source)
        if cell.cell_type == "markdown":
            heading = re.match(r"##\s+(.*)", src.strip())
            if heading:
                title = heading.group(1).split("(")[0].strip().rstrip("—-").strip()
                drop_section = any(title.startswith(d) for d in DROP_SECTIONS)
                keep_output = title.startswith(KEEP_OUTPUT_SECTION)
                if drop_section:
                    continue
                for old, new in RETITLE.items():
                    if title.startswith(old):
                        src = src.replace(heading.group(1), new, 1)
            elif drop_section:
                continue
            if src.strip().startswith("# "):        # the H1 becomes \title
                src = "\n".join(src.splitlines()[1:]).lstrip("\n")
            cell.source = _promote_headings(src)
            cells.append(cell)
            continue

        if drop_section:
            continue

        # A code cell contributes its filtered text output, then its figure -- never the
        # code itself, and never a raw dataframe.
        text = "\n".join(
            t for t in (_clean_stream("".join(o.get("text", "")))
                        for o in cell.get("outputs", []) if o.output_type == "stream")
            if t.strip()) if keep_output else ""
        has_image = any(o.output_type == "display_data"
                        and any(k.startswith("image") for k in o.get("data", {}))
                        for o in cell.get("outputs", []))
        if text.strip():
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=text)]
            cells.append(cell)
        if has_image:
            if n_fig >= len(PANELS):
                raise RuntimeError(f"more figure cells than PANELS ({len(PANELS)})")
            stem, key, name = PANELS[n_fig]
            n_fig += 1
            pdf = os.path.join(out_dir, f"{stem}{suffix}.pdf")
            if not os.path.exists(pdf):
                raise FileNotFoundError(
                    f"{pdf} -- run the notebook first, or pass -suffix for the other set")
            cells.append(nbformat.v4.new_raw_cell(
                _figure(pdf, _escape(captions[key]), name),
                metadata={"raw_mimetype": "text/latex", "format": "text/latex"}))

    if n_fig != len(PANELS):
        raise RuntimeError(f"{n_fig} figure cells but {len(PANELS)} panels named; the "
                           "notebook's figures changed -- update PANELS")

    nb.cells = cells
    # Checked per cell, not against the concatenation: a pattern ending at a cell
    # boundary matches the joined text and then silently fails to match any cell.
    for old, new in REWRITES:
        hits = sum(c.source.count(old) for c in nb.cells if c.cell_type == "markdown")
        if hits != 1:
            raise RuntimeError(f"REWRITES entry matched {hits} times: {old[:60]!r}")
    for c in nb.cells:
        if c.cell_type == "markdown":
            for old, new in REWRITES:
                c.source = c.source.replace(old, new)

    nb.metadata["title"] = TITLE
    nb.metadata["authors"] = [{"name": AUTHORS}]
    return nb


# Added to nbconvert's preamble. The `none` counter is not optional: pandoc emits
# `\def\LTcaptype{none}` for a caption-less markdown table, and the `caption` package
# nbconvert already loads then steps a counter by that name -- "No counter 'none'
# defined" is the whole build failing on the one table in the introduction. The rest is
# house style for a submitted supplement -- small justified captions, floats that stay
# near their text.
PREAMBLE = r"""
\usepackage{caption}
\makeatletter\@ifundefined{c@none}{\newcounter{none}}{}\makeatother
% format=plain is the load-bearing key: nbconvert's own preamble sets
% format=nocaption, which silently typesets every \caption as nothing.
\captionsetup{format=plain,font=small,labelfont=bf,justification=justified,
               singlelinecheck=false,skip=6pt}
\usepackage{float}
\renewcommand{\topfraction}{0.9}
\renewcommand{\floatpagefraction}{0.8}
"""


def export(nb, dest, keep_tex):
    from nbconvert import LatexExporter
    from traitlets.config import Config

    c = Config()
    c.LatexExporter.exclude_input = True
    c.LatexExporter.exclude_input_prompt = True
    c.LatexExporter.exclude_output_prompt = True
    c.LatexExporter.template_name = "latex"
    body, _ = LatexExporter(config=c).from_notebook_node(nb)
    body = body.replace(r"\begin{document}", PREAMBLE + "\n" + r"\begin{document}", 1)
    body = body.replace(r"\maketitle", "\\maketitle\n\\thispagestyle{empty}", 1)

    build_dir = os.path.join(os.path.dirname(dest), "_supporting_text_build")
    os.makedirs(build_dir, exist_ok=True)
    tex = os.path.join(build_dir, "supporting_text.tex")
    with open(tex, "w") as fh:
        fh.write(body)
    for _ in range(2):                      # twice: the ToC/refs need a second pass
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            os.path.basename(tex)],
                           cwd=build_dir, capture_output=True, text=True)
    if r.returncode != 0:
        log = os.path.join(build_dir, "supporting_text.log")
        tail = [l for l in open(log, errors="replace").read().splitlines()
                if l.startswith("!") or "Undefined" in l][:20]
        raise RuntimeError("xelatex failed:\n  " + "\n  ".join(tail) + f"\n  (log: {log})")
    shutil.copy(os.path.join(build_dir, "supporting_text.pdf"), dest)
    if keep_tex:
        shutil.copy(tex, os.path.splitext(dest)[0] + ".tex")
    else:
        shutil.rmtree(build_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-nb", default=os.path.join(HERE, "fig5.ipynb"))
    ap.add_argument("-out_dir", default=os.path.join(HERE, "output"))
    ap.add_argument("-suffix", default=config.WINDOW_SET_SUFFIX,
                    help="window set: '' or '.neutral' (default: what config.py is set to)")
    ap.add_argument("-keep_tex", action="store_true")
    args = ap.parse_args()

    nb = build(args.nb, args.out_dir, args.suffix, args.keep_tex)
    dest = os.path.join(args.out_dir, f"supporting_text{args.suffix}.pdf")
    export(nb, dest, args.keep_tex)
    print("wrote", dest)

    # Prose that reads as a notebook rather than as a document. Not an error -- the fix
    # belongs in make_fig5_nb.py, and it is a judgement call each time.
    isms = re.compile(r"the cell (below|above)|`fig5/|\.py`|this repo|notebook")
    for c in nb.cells:
        if c.cell_type == "markdown":
            for line in c.source.splitlines():
                if isms.search(line):
                    print("  notebook-ism:", line.strip()[:100])


if __name__ == "__main__":
    main()
