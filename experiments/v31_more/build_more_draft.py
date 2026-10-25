"""Build paper/drafts/more.tex: the OLD texts are cut from the current paper/main.tex by unique anchors, the NEW texts
are below.  Rerun after main.tex changes:  python experiments/v31_more/build_more_draft.py $(git rev-parse --short HEAD)
then python experiments/v31_more/apply_more_draft.py --check."""
import sys

import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
main = open(f'{ROOT}/paper/main.tex').read()


def cut(start, end=None):
    """Exact text of main.tex from `start` (inclusive) to the end of `end` (inclusive); start must be unique."""
    assert main.count(start) == 1, (start[:60], main.count(start))
    i = main.index(start)
    if end is None:
        return start
    j = main.index(end, i) + len(end)
    return main[i:j]


HEADER = r"""% =====================================================================================================
% Draft: follow-up checks #27 (coverage on real data with calibrated noise), #30 (the coverage repeat),
% #31 (about 50 encrypted runs per setting; 550 runs made) and #34 (cryptographic cost of full-record SSP).
%
% How to merge into paper/main.tex (mechanical):
%   Each block below runs from "%R-BEGIN <id>" to "%R-END <id>". The lines starting with "%OLD| " give the
%   exact old text of main.tex (strip the 6-character prefix "%OLD| "; "%OLD|" alone is an empty line; join
%   with newlines). Each old text occurs exactly once in main.tex as of commit __COMMIT__. Replace it by the
%   lines between "%NEW-BEGIN" and "%NEW-END", verbatim.
%       python experiments/v31_more/apply_more_draft.py --check                  # every old text occurs once
%       python experiments/v31_more/apply_more_draft.py --out /tmp/main_more.tex # writes a merged copy
%   The script never writes main.tex itself. New labels: tab:coverage_real, fig:coverage_real,
%   fig:coverage_repeat, fig:noise_more; new figures paper/figures/{coverage_real,coverage_repeat,noise}_more.png.
%   Check items #27, #30 and #31 are closed; #34 is replaced by a narrower item.
%
% Sources: reports/v31_more/README.md and key_numbers.md (experiments/v31_more/tables_more.py).
% =====================================================================================================
"""

blocks = []


def block(bid, desc, old, new):
    blocks.append((bid, desc, old, new.strip('\n')))


# ---------------------------------------------------------------------------------------------- abstract
block('M01', 'Abstract: coverage on real data',
      cut(r"On a synthetic population at $\delta = 10^{-9}$, the Wald intervals under setting I cover $94.5$--$96.6\%$ in every cell (target $95.5\%$), with and without the ridge."),
      r"""On a synthetic population at $\delta = 10^{-9}$, the Wald intervals under setting I cover $94.5$--$96.6\%$ in every cell (target $95.5\%$), with and without the ridge, and $94.5$--$96.4\%$ on fixed cohorts of five real datasets with the calibrated noise of a real budget.""")

block('M02', 'Abstract: the encrypted runs',
      cut(r"End-to-end runs of a full OpenFHE implementation reproduce a plaintext simulation of the mechanism under every setting evaluated."),
      r"""End-to-end runs of a full OpenFHE implementation reproduce a plaintext simulation of the mechanism under every setting evaluated; in $550$ runs under settings I and P the decrypted noise has its calibrated law, with mean zero to within $0.001$ of its standard deviation over $14.7$ million values.""")

# ---------------------------------------------------------------------------------------------- coverage subsection
block('M03', 'Section 13.4 (coverage): real data with calibrated noise; closes check #27',
      cut(r"\tocheck{The coverage table uses noise levels set directly (sigma-1 = 2.5 sigma-2).", "repeat the check on real data.}"),
      r"""\paragraph{Real data with calibrated noise.} Table~\ref{tab:coverage} sets the noise levels directly. Table~\ref{tab:coverage_real} and Figure~\ref{fig:coverage_real} repeat the check for the Wald interval on real data with the noise of a real budget: setting I, $\sigma_1$ and $\sigma_2$ from Calibrate at $\delta = 10^{-9}$ and $\varepsilon \in \{1, 4, 8\}$, and block normalization with the reference rows of fold~0 (Section~\ref{sec:eval:adassp}). The data are fixed. The cohort is the first $n$ matched rows of fold~0, or all of them, so $\btheta_\lambda$ is a fixed function of the data and of each release's $\lambda$. Each cell has $1{,}000$ independent releases of the whole mechanism, with the implementation's bootstrap ($2{,}000$ antithetic pairs). Coefficients of one release share its noise, so each interval in the table is clustered by release.
\begin{itemize}[noitemsep]
  \item On protein, bike, elevators and pol, block normalization keeps the ridge on in every release at every size these datasets allow. Protein's $n_{\mathrm{off}}$ is about $80$k to $560$k for $\varepsilon = 8$ to $1$, and the block-normalized designs of bike and elevators are singular. We add 3droad ($p = 4$), where the ridge is off from $n = 1{,}000$ at $\varepsilon \ge 4$ and from $5{,}000$ at $\varepsilon = 1$, so that both regimes are checked on real data.
  \item The Wald interval covers $\btheta_\lambda$ $0.945$--$0.964$ of the time over the $48$ cells, and single coefficients $0.935$--$0.979$. On protein it covers $0.952$--$0.963$ in all twelve cells, and every protein interval contains the target or lies above it. The gate refused no release.
  \item While the ridge is on, the interval leans conservative: $26$ of the $40$ ridge-on cells lie above the target (elevators $0.959$--$0.964$). One cell lies below: 3droad at $n = 25{,}000$, $\varepsilon = 1$ ($0.945$). Rerun with $5{,}000$ fresh releases, it gives $0.955$ ($0.952$--$0.959$).
  \item Coverage of $\bbeta$ cannot be checked this way. With a fixed cohort, coverage of the full-data least-squares fit (the reference of Section~\ref{sec:eval:fws}) is conditional on one sample, whereas the bias-aware interval's guarantee holds over the sampling and the noise together. Where the mechanism noise is small against the sampling error, each coefficient is covered in almost every release or in almost none (3droad at $n = 25{,}000$: $0.67$--$0.85$ over three slopes). While $\hat L = 0$ the interval is unbounded anyway. Section~\ref{sec:eval:fws} checks the intervals for $\bbeta$ on real data with resampled cohorts.
\end{itemize}

\begin{table}[ht]
  \centering\small
  \caption{Real data, block normalization, setting I, calibrated noise ($\delta = 10^{-9}$), fixed cohort of the first $n$ matched rows of fold~0 (the last $n$ of each dataset but 3droad is all of them): coverage of $\btheta_\lambda$ by the Wald interval over all coefficients, $1{,}000$ releases per cell, with a release-clustered $95\%$ interval. Target $0.955$. Ridge: on or off in every release of the cell (3droad at $n = 1{,}000$: on at $\varepsilon = 1$, off at $4$ and $8$). $^{a}$Rerun with $5{,}000$ fresh releases: $0.955$ ($0.952$--$0.959$).}
  \label{tab:coverage_real}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lrrlccc@{}}
    \toprule
    dataset & $p$ & $n$ & ridge & $\varepsilon = 1$ & $\varepsilon = 4$ & $\varepsilon = 8$ \\
    \midrule
    protein & 10 & 1{,}000 & on & $0.962$ (0.958--0.966) & $0.960$ (0.955--0.964) & $0.959$ (0.954--0.963) \\
     & & 5{,}000 & on & $0.963$ (0.960--0.967) & $0.961$ (0.957--0.965) & $0.955$ (0.950--0.960) \\
     & & 20{,}000 & on & $0.958$ (0.953--0.962) & $0.959$ (0.954--0.964) & $0.952$ (0.947--0.958) \\
     & & 36{,}584 & on & $0.958$ (0.953--0.962) & $0.958$ (0.952--0.963) & $0.953$ (0.947--0.959) \\
    \addlinespace
    bike & 18 & 1{,}000 & on & $0.962$ (0.959--0.965) & $0.961$ (0.958--0.964) & $0.959$ (0.956--0.962) \\
     & & 5{,}000 & on & $0.960$ (0.957--0.963) & $0.959$ (0.955--0.962) & $0.956$ (0.952--0.959) \\
     & & 13{,}903 & on & $0.958$ (0.955--0.962) & $0.959$ (0.955--0.962) & $0.956$ (0.952--0.959) \\
    \addlinespace
    elevators & 19 & 1{,}000 & on & $0.963$ (0.960--0.966) & $0.963$ (0.960--0.966) & $0.962$ (0.959--0.965) \\
     & & 5{,}000 & on & $0.961$ (0.958--0.964) & $0.961$ (0.958--0.964) & $0.961$ (0.958--0.964) \\
     & & 13{,}279 & on & $0.964$ (0.961--0.967) & $0.962$ (0.959--0.965) & $0.959$ (0.956--0.962) \\
    \addlinespace
    pol & 27 & 1{,}000 & on & $0.961$ (0.958--0.963) & $0.960$ (0.958--0.962) & $0.958$ (0.956--0.961) \\
     & & 5{,}000 & on & $0.958$ (0.956--0.961) & $0.960$ (0.958--0.963) & $0.956$ (0.953--0.959) \\
     & & 12{,}000 & on & $0.960$ (0.958--0.963) & $0.957$ (0.954--0.960) & $0.959$ (0.956--0.962) \\
    \addlinespace
    3droad & 4 & 1{,}000 & on/off & $0.955$ (0.949--0.961) & $0.961$ (0.955--0.968) & $0.951$ (0.944--0.959) \\
     & & 5{,}000 & off & $0.959$ (0.953--0.966) & $0.960$ (0.953--0.967) & $0.954$ (0.947--0.961) \\
     & & 25{,}000 & off & $0.945^{a}$ (0.937--0.952) & $0.956$ (0.949--0.963) & $0.956$ (0.949--0.963) \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/coverage_real_more.png}
  \caption{The coverage of Table~\ref{tab:coverage_real}: Wald coverage of $\btheta_\lambda$ under setting I with calibrated noise on fixed real cohorts, with release-clustered $95\%$ intervals; colour: $\varepsilon$; hollow: the ridge is on. Grey diamond: the rerun of the one cell whose interval lies below the target. Horizontal line: the target $0.955$.}
  \label{fig:coverage_real}
\end{figure}""")

# ---------------------------------------------------------------------------------------------- e2e: protein
block('M04', 'Section 13.5 (end-to-end runs, protein): the coverage repeat in plaintext',
      cut(r"Pooled over all 30 runs, the Wald interval covers $\btheta_\lambda$ for $286$ of $300$ coefficients (target $0.955$)."),
      r"""Pooled over all 30 runs, the Wald interval covers $\btheta_\lambda$ for $286$ of $300$ coefficients (target $0.955$). A plaintext simulation of the same six configurations, with the noise levels of the encrypted runs and $10{,}000$ releases each, covers $0.955$--$0.958$, with release-clustered $95\%$ intervals inside $0.953$--$0.960$. The encrypted counts, $46$ to $50$ of $50$ per configuration, are what batches of five runs give: a plaintext batch covers no more in $21$--$100\%$ of draws (Figure~\ref{fig:coverage_repeat}).""")

# ---------------------------------------------------------------------------------------------- e2e: synthetic ridge off
block('M05', 'Section 13.5 (end-to-end runs, synthetic, ridge off): closes check #30',
      cut(r"\tocheck{Coverage at epsilon = 8 (0.927) and for some protein settings (46 of 50) is slightly below 0.955.", "before drawing conclusions.}"),
      r"""A plaintext simulation of these two configurations on the same data, with the noise levels of the encrypted runs and $10{,}000$ releases each, covers $0.957$ ($0.956$--$0.958$, release-clustered) at $\varepsilon = 4$ and $0.955$ ($0.954$--$0.957$) at $\varepsilon = 8$. Single coefficients cover $0.951$--$0.961$, and the standardized errors have s.d.\ $0.993$ and $0.999$. So the coverage at $\varepsilon = 8$ is on target, and $204$ of $220$ is a low draw: a plaintext batch of 20 runs covers no more with probability $0.05$. Pooled over the eight configurations of this section, the encrypted runs covered $699$ of $740$ coefficients ($0.945$) against $707$ expected from the simulations (s.d.\ $6.1$; one-sided $p = 0.08$). The runs of Section~\ref{sec:eval:sim} test the encrypted coverage with many more runs.

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/coverage_repeat_more.png}
  \caption{Wald coverage of $\btheta_\lambda$ in the configurations of the end-to-end runs of this section (previous setting): a plaintext simulation of the same mechanism on the same data with the noise levels of the encrypted runs ($10{,}000$ releases, release-clustered $95\%$ interval) against the encrypted runs (20 per synthetic and 5 per protein configuration; Wilson $95\%$ interval, which ignores the clustering and is therefore slightly too narrow). Horizontal line: the target $0.955$.}
  \label{fig:coverage_repeat}
\end{figure}""")

# ---------------------------------------------------------------------------------------------- encrypted check of I and P
block('M06', 'Section 13.6 (coefficient accuracy): the encrypted check of settings I and P, 550 runs; closes check #31',
      cut(r"\paragraph{Encrypted check of settings I and P.} We ran the full OpenFHE implementation fifteen times",
          "and recheck these means before relying on the runs.}"),
      r"""\paragraph{Encrypted check of settings I and P.} We ran the full OpenFHE implementation $550$ times on one synthetic dataset from this population: $n = 10{,}000$ matched records in a register of $12{,}000$ rows padded to $16{,}384$, $\delta = 10^{-9}$. There were three configurations. In two of them the ridge is on: setting I at $\varepsilon = 2$ and setting P at $\varepsilon = 1$, where the matched design's $n_{\mathrm{off}}$ is $14.1$k and $16.4$k. In the third it is off: setting I at $\varepsilon = 8$, where $n_{\mathrm{off}}$ is $3.9$k. The flooding parameter was $\kappa = 40$, $40$ and $48$. Each configuration got $100$ runs. Because I at $\varepsilon = 2$ came out low (below), $150$ more runs of it followed with fresh seeds. A run took $20$--$23$\,s (median) on one core per party. $\Rp$ read the exact radius from the cached quantile: $r_1 = 3{,}419$ under I at $\varepsilon = 2$ and $3{,}964$ under P.
\begin{itemize}[noitemsep]
  \item No release was refused, and $\bA$ was exact. We rebuilt each run's decrypted releases from its recorded noise and fed them to the plaintext solver with the run's seed. This gave the run's $\lambda$, estimate and half-widths to within $10^{-11}$ (relative), so an encrypted run differs from a plaintext one only through its noise.
  \item The ridge lay inside the $95\%$ range of a plaintext simulation of the same mechanism on the same data (4{,}000 draws). Under I at $\varepsilon = 2$ it did so in $95$ of $100$ runs and $144$ of $150$ runs of the repeat (range $2{,}578$--$4{,}232$). Under P it did so in $92$ of $100$ runs ($4{,}837$--$7{,}855$). Under I at $\varepsilon = 8$ it was $0$ in every run, as simulated. The encrypted coefficients lay inside the simulated $95\%$ ranges in $0.940$, $0.955$, $0.944$ and $0.957$ of cases (I at $\varepsilon = 2$, its repeat, P, I at $\varepsilon = 8$).
  \item Under I at $\varepsilon = 2$, the Wald interval covered $\btheta_\lambda$ for $1{,}037$ of $1{,}100$ coefficients ($0.943$; release-clustered $95\%$ interval $0.928$--$0.958$). The repeat covered $1{,}586$ of $1{,}650$ ($0.961$; $0.953$--$0.970$), and the two together $0.954$ ($0.946$--$0.962$). Under I at $\varepsilon = 8$ it covered $1{,}054$ of $1{,}100$ ($0.958$; $0.945$--$0.971$). The plaintext simulation of the same configurations covers $0.9585$ and $0.9555$ ($10{,}000$ releases each). The first $100$ runs at $\varepsilon = 2$ lie just below the $2.5\%$ quantile of the plaintext simulation for 100 runs ($1{,}040$). The repeat and the noise checks below put this down to chance.
  \item The raw decrypted noise was standardized by the calibrated $\sigma_1$ and $\sigma_2$: in stage~1, the $51$ $\Op$-dependent entries of $\bG$; in stage~2, those, $\bc$ and $y^\top y$. Pooled over all $450$ runs, the stage-1 noise has mean $+0.006$ (standard error $0.007$) and s.d.\ $0.995$ (Kolmogorov--Smirnov $p = 0.13$ against the calibrated law, $\mathcal N(0, 1)$ after standardization). The stage-2 noise has mean $+0.004$ ($0.006$) and s.d.\ $1.006$ ($p = 0.48$). For each configuration and release, every mean lies within $1.9$ standard errors of zero. The slots outside the packed entries carry the same noise on a plaintext zero. There are more than $1.6$ million of them per configuration and release. Their mean lies within $0.0011$ of zero (standard error $0.0006$--$0.0008$), and their s.d.\ is $0.999$--$1.001$, release by release. Over all $14.7$ million slot values the mean is $-0.0004$ (standard error $0.0003$).
  \item The negative means of the earlier five-run checks are not reproduced: $-0.17$ on the stage-1 noise under I, and $-0.10$ under P (Figure~\ref{fig:noise_more}). The same configurations now give $+0.03$ and $-0.001$ on the stage-1 noise under I ($5{,}100$ and $7{,}650$ entries), and $+0.006$ and $-0.003$ under P.
\end{itemize}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/noise_more.png}
  \caption{Raw decrypted noise of $450$ encrypted runs (settings I and P, $n = 10{,}000$, register $16{,}384$), standardized by the calibrated $\sigma_1$ and $\sigma_2$. Left: histogram of all released entries against $\mathcal N(0, 1)$. Middle: quantile plot. Right: mean with $95\%$ interval by configuration (I2: setting I at $\varepsilon = 2$; I2r: its repeat; P1: setting P at $\varepsilon = 1$; I8: setting I at $\varepsilon = 8$) and release, pooled, and for the earlier five-run checks (hollow).}
  \label{fig:noise_more}
\end{figure}""")

# ---------------------------------------------------------------------------------------------- full-record cost
block('M07', 'Section 13.7 (baselines, full-record SSP results): its cryptographic cost',
      cut(r"  \item \emph{Against AdaSSP} the verdicts do not change: under block normalization and setting I full-record SSP is worse than AdaSSP (published) on $20$ of $29$ datasets at $\varepsilon = 1$ with a median gap of $+26\%$, against $+25\%$ for v3.1."),
      r"""  \item \emph{Against AdaSSP} the verdicts do not change: under block normalization and setting I full-record SSP is worse than AdaSSP (published) on $20$ of $29$ datasets at $\varepsilon = 1$ with a median gap of $+26\%$, against $+25\%$ for v3.1.
  \item \emph{Cryptographic cost} (from the implementation's cost model, single thread, $d_R = 4$, $d_O = 6$; nothing was run). The releases also hold $\triu\bA$ without $n$: $(d_R + 1)(d_R + 2)/2 - 1 = 14$ more entries. The four sums $\sum_t b_t x_{R,j,t}$ come from $\Rp$'s existing uploads. The ten products need ten more encrypted upload vectors $b \cdot x_j x_k$ from $\Rp$, on top of the six of v3.1. $\Op$'s aggregation grows by $77/63 = 1.22$: $3.6$, $3.8$ and $4.7$\,s more than v3.1's $16.3$, $17.0$ and $21.3$\,s at $4$k, $16$k and $64$k register rows. $\Op$'s CPU time grows by $13$--$17\%$ and the upload by $5$, $10$ and $40$\,MB (communication $\times 1.10$, $1.19$ and $1.55$). The releases cost the same, since each is one ciphertext whose noise is sampled per ring coefficient. With the larger entry counts, Calibrate raises $\log_2 B_{\mathrm{tot}}$ by $0.15$--$0.18$, with the same $\kappa$ and $\sigma$.""")

block('M08', 'Section 13.7 (baselines): check #34 narrowed',
      cut(r"\tocheck{Full-record SSP was simulated in plaintext only.", "would have its own cost.}"),
      r"""\tocheck{Full-record SSP: its cryptographic cost is computed from the cost model, not measured, and assumes that R uploads the products b x-j x-k; the U-side bound of those vectors (at most B-R squared) enters the derivation of B-tot, which was only re-evaluated with the larger entry counts. A secure-aggregation protocol of the kind of Karr et al. or Gascon et al. would have its own cost.}""")

# ---------------------------------------------------------------------------------------------- summary
block('M09', 'Section 15 (remaining obligations): coverage on real data',
      cut(r"and still to be checked on real data under v3.1."),
      r"""and, for the Wald interval, also on fixed cohorts of five real datasets with calibrated noise (Section~\ref{sec:eval:coverage}); on real data the intervals for $\bbeta$ are checked only against a full-data reference (Section~\ref{sec:eval:fws}).""")

commit = sys.argv[1] if len(sys.argv) > 1 else 'HEAD'
out = [HEADER.replace('__COMMIT__', commit)]
for bid, desc, old, new in blocks:
    out.append(f'%R-BEGIN {bid} {desc}')
    for line in old.split('\n'):
        out.append('%OLD| ' + line if line else '%OLD|')
    out.append('%NEW-BEGIN')
    out.append(new)
    out.append('%NEW-END')
    out.append(f'%R-END {bid}')
    out.append('')
open(f'{ROOT}/paper/drafts/more.tex', 'w').write('\n'.join(out))
print('wrote paper/drafts/more.tex with', len(blocks), 'blocks')
