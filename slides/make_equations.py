"""Typeset the deck's display equations with LaTeX.

Reads slides/equations.json ({key: {tex, color}}) and writes, for each key,
slides/assets/eq/<key>.png (transparent, 600 dpi) plus
slides/assets/eq/manifest.json with each image's size in points at 10 pt.
build_deck.js scales the images from that manifest, so it does not need LaTeX.

Needs pdflatex and pdftocairo (poppler). Run from the repo root:
    python slides/make_equations.py
"""
import json
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "assets" / "eq"
DPI = 600

TEMPLATE = r"""\documentclass[10pt,border=1pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage{xcolor}
\begin{document}
\color[HTML]{%(color)s}$\displaystyle %(tex)s$
\end{document}
"""


def page_size_pt(pdf):
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    w, h = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info).groups()
    return float(w), float(h)


def main():
    eqs = json.loads((ROOT / "equations.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for key, spec in eqs.items():
            (tmp / f"{key}.tex").write_text(TEMPLATE % spec)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{key}.tex"],
                           cwd=tmp, capture_output=True, check=True)
            subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-r", str(DPI),
                            str(tmp / f"{key}.pdf"), str(OUT / key)], check=True)
            w, h = page_size_pt(tmp / f"{key}.pdf")
            manifest[key] = {"file": f"{key}.png", "w_pt": round(w, 3), "h_pt": round(h, 3)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} equations to {OUT}")


if __name__ == "__main__":
    main()
