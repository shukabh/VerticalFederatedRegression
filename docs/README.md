# Documents: specification and proofs of protocol v3.1

These are the specification of protocol v3.1 and its proof documents. The paper (`paper/main.tex`) cites them by
path. Each document is committed as LaTeX source together with its built PDF.

| file | what it is | pages |
|---|---|---|
| `protocol_v31.tex` / `.pdf` | The specification of protocol v3.1: threat model and goals, the changes from v3, notation, the algorithms (Setup, Link, Upload, Aggregate, Release, Solve and infer), parameters, the ledger, and the extensions. | 14 |
| `protocol_v31_fig.tex` / `.pdf` | The protocol figure (TikZ), included by the specification. | 1 |
| `proofs/crypto_proofs.tex` / `.pdf` (with `proofs/crypto/`) | Part A: cryptographic and DP guarantees. Encoding sensitivity, smudging, the worst-case error bound B_tot, decoupled scales and OpenFHE's key switching, the decoding offset, output privacy, input and cohort privacy (G1, G2), disclosure (G4) and the cuckoo-hashing slack, correctness, the soundness of Calibrate, and corrections to the specification. | 36 |
| `proofs/stats_proofs.tex` / `.pdf` (with `proofs/stats_s7_s10.tex`) | Part B: sensitivity and statistical guarantees. The replace-one sensitivity, operator-norm concentration, the two-stage choice of the ridge, the bound on the coefficients, coverage of both intervals, and the bootstrap RSS correction (Sections S1-S10), including the exact full-record sensitivity (S10c, S10d) and the full-record circuit's B_tot, centre sensitivity and offset (S10e). | 30 |
| `proofs/composition/composition_proof.tex` / `.pdf` | Part C: composition across runs for an adaptive, possibly colluding coalition of researchers, with a self-contained Renyi filter, and the hockey-stick treatment of the smudging step that the ledger uses. | 21 |
| `proofs/protocol_v31_proofs.tex` / `.pdf` | The proofs appendix: an overview, a status table of every claim and the open items, followed by Parts A, B and C, included as PDFs. | 91 |

The scripts that the proofs cite for their numerical checks (`scratch/`, `scratch_stats/`, `scripts/kappa_numbers.py`)
are not included here. The checks that the repository keeps are in `experiments/proofs_check/`.

## Building

Each document builds from its own folder with relative paths, with `pdflatex` and no BibTeX (the bibliographies are
inline). The proofs appendix includes the PDFs of the three parts, so build the parts first.

```bash
cd docs
pdflatex protocol_v31_fig.tex                 # the figure, once
pdflatex protocol_v31.tex && pdflatex protocol_v31.tex

cd proofs
for f in crypto_proofs stats_proofs; do pdflatex $f.tex && pdflatex $f.tex && pdflatex $f.tex; done
(cd composition && pdflatex composition_proof.tex && pdflatex composition_proof.tex)
pdflatex protocol_v31_proofs.tex && pdflatex protocol_v31_proofs.tex
```

Packages used: amsmath, amssymb, amsthm, mathtools, geometry, booktabs, array, tabularx, longtable, xltabular,
enumitem, xcolor, microtype, algorithm, algpseudocode, hyperref, graphicx, pdfpages and tikz-cd (all in TeX Live).
The build artifacts (`.aux`, `.log`, `.out`, `.toc`) are ignored by git.
