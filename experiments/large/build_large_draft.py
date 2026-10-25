"""Build paper/drafts/large.tex: the OLD texts are cut from the current paper/main.tex by unique anchors, the NEW texts
are below (numbers from reports/large/README.md and key_numbers.md).  Rerun after main.tex changes:
    python experiments/large/build_large_draft.py $(git rev-parse --short HEAD)
    python experiments/large/apply_large_draft.py --check
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
main = open(f'{ROOT}/paper/main.tex').read()


def cut(start, end=None):
    """Exact text of main.tex from `start` (inclusive) to the end of `end` (inclusive); start must be unique."""
    assert main.count(start) == 1, (start[:70], main.count(start))
    if end is None:
        return start
    i = main.index(start)
    j = main.index(end, i) + len(end)
    return main[i:j]


HEADER = r"""% =====================================================================================================
% Draft: the large-scale evaluation (register sizes 100k, 300k and 1M rows; regime B with n_R^max = 100,000;
% 3droad) and the 64-bit PSI items (two felts per item).
%
% How to merge into paper/main.tex (mechanical):
%   Each block below runs from "%R-BEGIN <id>" to "%R-END <id>". The lines starting with "%OLD| " give the
%   exact old text of main.tex (strip the 6-character prefix "%OLD| "; "%OLD|" alone is an empty line; join
%   with newlines). Each old text occurs exactly once in main.tex as of commit __COMMIT__. Replace it by the
%   lines between "%NEW-BEGIN" and "%NEW-END", verbatim.
%       python experiments/large/apply_large_draft.py --check                   # every old text occurs once
%       python experiments/large/apply_large_draft.py --out /tmp/main_large.tex # writes a merged copy
%   The script never writes main.tex itself. New labels: sec:eval:large, tab:cost_large, fig:cost_large,
%   fig:accuracy_large, fig:noise_large; new figures paper/figures/{cost,accuracy,noise}_large.png.
%   The check item on the staged benchmarks is removed.
%
% Sources: reports/large/README.md, key_numbers.md, stage{1..5}.json (experiments/large/).
% =====================================================================================================
"""

blocks = []


def block(bid, desc, old, new):
    blocks.append((bid, desc, old, new.strip('\n')))


# ---------------------------------------------------------------------------------------------- abstract
old = cut(r"in $450$ runs under settings I and P the decrypted noise has its calibrated law, with mean zero to within $0.001$ of its standard deviation over $14.7$ million values.")
block('L01', 'Abstract: the measured cost at one million rows', old, old + r""" At $10^6$ register rows and $20{,}000$ padded queries a full run takes $3.0$ minutes on a 4-core machine ($\Op$: $152$\,s and $2.3$\,GB; $475$\,MB of traffic), with the linkage exact in all $78$ large runs.""")

# ---------------------------------------------------------------------------------------------- parameters table
old = cut(r"$N_B$, $t$ & BFV ring dimension; plaintext modulus, which is the item length &", r"which does not meet the constraint (Appendix~\ref{app:psi}) \\")
block('L02', 'Parameters table: the item length is F felts of Z_t', old, r"""$N_B$, $t$, $F$ & BFV ring dimension; plaintext modulus; field elements per item (the item length is $F\log_2(t-1)$) & $t$ prime, $t \equiv 1 \bmod 2N_B$; $p_{\mathrm{pass}} \le \delta_{\mathrm{psi}}$ for false matches (Appendix~\ref{app:psi}), checked at Setup & $2^{14}$; $t \approx 2^{32}$; $F = 2$ (64-bit items) \\""")

# ---------------------------------------------------------------------------------------------- G1 cuckoo
old = cut(r"the failure probability of cuckoo hashing (at most $2^{-49.1}$ with $n_R^{\max} = 2{,}000$ items,", r"proofs, Part~A, Lemma~7.5)")
block('L03', 'G1: cuckoo slack with balanced tables at larger n_R^max', old, r"""the failure probability of cuckoo hashing (at most $2^{-49.1}$ with $n_R^{\max} = 2{,}000$ items, $h = 3$ hash functions and $2^{14}$ bins, and at most $2^{-40}$ up to a load of $0.64$, for an insertion that finds an assignment whenever one exists, as the implementation's random walk with an augmenting-path fallback does; proofs, Part~A, Lemma~7.5; above $10{,}472$ items $\Rp$ splits its queries into the fewest balanced tables whose summed bound is at most $\delta_{\mathrm{psi}}$: two tables and $2^{-43.4}$ at $20{,}000$, ten and $2^{-41.1}$ at $100{,}000$)""")

# ---------------------------------------------------------------------------------------------- appendix: items
old = cut(r"Each row $j$ becomes the item $F_k(\mathrm{id}_j)$, mapped into $\Z_t$ so as to avoid the padding root $t - 1$, with label $\tau(j)$ encrypted under a key derived from $F'_k(\mathrm{id}_j)$.")
block('L04', 'Appendix PSI: items and labels are F felts', old, r"""Each row $j$ becomes the item $F_k(\mathrm{id}_j)$, mapped into $\Z_t^F$ as $F$ field elements (\emph{felts}, as in APSI) that each avoid the padding root $t - 1$, with label $\tau(j)$ encrypted under $F$ independent keys derived from $F'_k(\mathrm{id}_j)$, one per felt ($F = 2$ in the evaluation, so items have $64$ bits). Felt $f$ of a bin occupies slot $f N_{\mathrm{bins}} + b$, that is, ciphertext $f$; each slot has its own membership and label polynomial, and the items of a partition differ in every felt.""")

# ---------------------------------------------------------------------------------------------- appendix: recovery
old = cut(r"\paragraph{Recovery at $\Rp$.} For each occupied bin and partition, $\Rp$ opens the label under $F'_k$ of its item when the flag is zero, which gives $\pi(i) = \tau(j)$.")
block('L05', 'Appendix PSI: the match rule with felts', old, r"""\paragraph{Recovery at $\Rp$.} For each occupied bin and partition, $\Rp$ declares a match only if the flag is zero in all $F$ felt slots of the bin \emph{and} the $F$ label felts, each opened under its own key derived from $F'_k$ of $\Rp$'s item, give the same position; that position is $\pi(i) = \tau(j)$. A true match always passes both tests. Flags that are zero in every felt but whose labels disagree come from felts matched by different items of the partition, which the zero test alone allows, and $\Rp$ treats them as no match, without an abort.""")

# ---------------------------------------------------------------------------------------------- appendix: collision remark
old = cut(r"A false match requires two distinct identifiers to share an item value inside a common bin.", r"so longer items are needed.")
block('L06', 'Appendix PSI: collision probability with F felts and the consistency test', old, r"""A false match requires a query and the items of one partition of its bin to share all $F$ felt values, possibly from different items. Felts are uniform on $t - 1$ values (up to the bias of reducing a $256$-bit hash, which the implementation includes), independent across felts, and the items of a partition differ in every felt, so with partition loads $c_a \le D$ summing to at most the public load bound $B$,
\[
  p_{\mathrm{felt}} = \frac{n_R^{\max}\, S_F}{(t - 1)^F}, \qquad S_F = \lfloor B/D \rfloor D^F + (B \bmod D)^F,
\]
bounds the probability that some query has all $F$ flags zero in some partition. Such an event passes the label test only if $F - 1$ independent keys of the query hit given values, and only an event that passes can abort the run (ambiguous, out of range, or repeated position) or change the output:
\[
  p_{\mathrm{pass}} = p_{\mathrm{felt}}\, t^{-(F - 1)}, \qquad p_{\mathrm{silent}} = p_{\mathrm{pass}}\, \bar n_O / t .
\]
$p_{\mathrm{pass}}$ is the false-match term $p_{\mathrm{false}}$ of the slack of G1 and G2 (Section~\ref{sec:input}), and \textsc{Setup} refuses unless $p_{\mathrm{pass}} \le \delta_{\mathrm{psi}}$. With $F = 1$ there is no label test, $p_{\mathrm{pass}} = p_{\mathrm{felt}}$, and $t \approx 2^{32}$ fails the constraint at every evaluation size ($p_{\mathrm{pass}} = 2^{-18.8}$ at $\bar n_O = 4{,}096$, $n_R^{\max} = 600$; the earlier per-item estimate $n_R^{\max}\bar n_O h/(N_{\mathrm{bins}}(t-1))$ gave $8.5 \times 10^{-5}$ at $10^6$ and $n_R^{\max} = 2{,}000$). The evaluation uses $F = 2$, $N_{\mathrm{bins}} = 2^{14}$, $h = 3$, $D = 64$. At $\bar n_O = 10^6$ ($B = 308$, $\alpha = 5$) and $n_R^{\max} = 20{,}000$: $p_{\mathrm{felt}} = 2^{-35.5}$, which alone exceeds $\delta_{\mathrm{psi}} = 2^{-40}$ and is why the label test is part of the match rule; $p_{\mathrm{pass}} = 2^{-67.5}$ and $p_{\mathrm{silent}} = 2^{-79.6}$. The per-item formula with $F$ felts, $n_R^{\max}\bar n_O h/(N_{\mathrm{bins}}(t-1)^2)$, which counts only collisions of a whole item, gives $2^{-42.2}$. At $n_R^{\max} = 100{,}000$, $p_{\mathrm{pass}} = 2^{-65.2}$. Two felts double the query, the powers and the replies per cuckoo table at the same $N_{\mathrm{bins}}$; \textsc{SanitizeBFV}'s conditions are unchanged, since every reply ciphertext is as before.""")

# ---------------------------------------------------------------------------------------------- cost section
old = cut(r"\paragraph{Measured and extrapolated cost.}", r"\tocheck{Run the staged benchmarks at 100k, 300k and 1M rows once the PSI item length and the datasets are decided.}")
block('L07', 'Cost: measured cost at 100k, 300k and 1M rows replaces the extrapolation; the check item is removed', old, r"""\paragraph{Measured cost.} We ran the full implementation at $\bar n_O = 106{,}496$, $303{,}104$ and $1{,}007{,}616$ register rows with $n_R^{\max} = 20{,}000$ padded queries ($p = 11$, two felts per PSI item, four OpenMP threads per party and four aggregation entries in parallel at $\Op$, on a 4-core 2.8\,GHz Xeon with 15\,GB of memory), and at $10^6$ rows with $n_R^{\max} = 100{,}000$ (Section~\ref{sec:eval:large}). Table~\ref{tab:cost_large} gives the cost of one run; the communication equals the model's exact count. At $10^6$ rows, $\Op$ spends $80$ of its $152$\,s on the OPRF evaluation of its whole register (fresh key every run, single-threaded), $19$\,s on PSI binning and interpolation, $18$\,s on the CKKS aggregation, $15$\,s on the PSI evaluation and $12$\,s on \textsc{SanitizeBFV}; $369$ of the $475$\,MB of traffic is $\Rp$'s upload. With $100{,}000$ queries the PSI evaluation and sanitization grow with the ten cuckoo tables ($74$ and $62$\,s) and the traffic to $756$\,MB.

\begin{table}[ht]
\centering\small
\caption{Measured cost of one run (setting I, $\varepsilon = 1$, mean of 3 runs; four threads per party), against the cost model with its published inputs and with inputs re-measured on the idle machine. Wall time in seconds.}
\label{tab:cost_large}
\begin{tabular}{@{}lrrrr@{}}
\toprule
$\bar n_O$ & $106{,}496$ & $303{,}104$ & $1{,}007{,}616$ & $1{,}007{,}616$ \\
$n_R^{\max}$ & $20{,}000$ & $20{,}000$ & $20{,}000$ & $100{,}000$ \\
\midrule
$\Op$, measured & $34.5$ & $56.6$ & $151.9$ & $270.5$ \\
$\Op$, model (published inputs) & $50.2$ & $77.0$ & $193.9$ & $357.3$ \\
$\Op$, model (re-measured inputs) & $29.0$ & $49.2$ & $130.1$ & $225.0$ \\
run, end to end & $42$ & $67$ & $179$ & $313$ \\
$\Op$'s CPU time, all threads & $78$ & $109$ & $256$ & $550$ \\
peak memory $\Op$ / $\Rp$ (GB) & $0.76$ / $0.36$ & $0.99$ / $0.66$ & $2.33$ / $1.73$ & $2.34$ / $1.79$ \\
traffic (MB) & $133$ & $205$ & $475$ & $756$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/cost_large.png}
  \caption{Left: $\Op$'s wall time per phase at $10^5$, $3 \times 10^5$ and $10^6$ register rows ($n_R^{\max} = 20{,}000$, setting I, $\varepsilon = 1$, four threads), against the cost model with its published inputs (filled diamonds) and with re-measured inputs (open triangles). Right: traffic per run.}
  \label{fig:cost_large}
\end{figure}

The cost model, built from per-operation microbenchmarks and exact operation counts, missed in two ways (Figure~\ref{fig:cost_large}). With its published inputs it overestimates $\Op$'s total by $25$--$52\%$ at every size, and single phases by up to a factor $2.8$, which fails the $25\%$ agreement we had required before running. The microbenchmarks behind it run $1.4$--$2.2$ times faster on the idle machine than in the published files; its interpolation term assumes full partitions, whereas the fill is $0.15$ to $0.58$; and its aggregation speedup was measured with one thread per entry rather than the four threads per entry of these runs ($3.96$ against a measured $3.13$). With re-measured inputs and the measured fill it underestimates $\Op$'s total by $10$--$25\%$, and a few small phases by up to $36\%$: the PSI interpolation and evaluation in two cells at $300$k, and the upload conversion at $10^6$ ($4.8$\,s against $3.6$\,s), which the published model had right. The earlier extrapolation ($260$\,s of $\Op$'s CPU on one core and $76$\,s on four cores at $10^6$) assumed one felt, $2{,}000$ queries and a parallel OPRF evaluation, which is not implemented; parallelizing it would remove most of the remaining $80$\,s.""")

# ---------------------------------------------------------------------------------------------- new evaluation subsection
old = cut(r"\subsection{Coefficient accuracy against intersection size}")
block('L08', 'New evaluation subsection: the large-scale runs', old, r"""\subsection{Large-scale runs}
\label{sec:eval:large}

We ran the full implementation $78$ times at register sizes from $10^5$ to $10^6$ rows, in three regimes, each cell three times on one fixed dataset ($\delta = 10^{-9}$, $\kappa$ derived, a fresh single-run ledger per run, two felts per PSI item; Section~\ref{sec:cost} gives the cost). (A) The register grows and the cohort is fixed: $\bar n_O = 106{,}496$, $303{,}104$ and $1{,}007{,}616$ ($10^5$, $3 \times 10^5$ and $10^6$ rows of the synthetic population, padded), $n_R^{\max} = 20{,}000$ with $12{,}000$ matched records, settings I and P on the standardized population and P on the block-normalized one, $\varepsilon \in \{1, 4\}$. (B) A large matched set: $\bar n_O = 1{,}007{,}616$, $n_R^{\max} = 100{,}000$ with $60{,}000$ or $95{,}000$ matched records, setting I, $\varepsilon \in \{0.5, 1\}$. (C) 3droad: $\Op$'s register is its $304{,}413$ rows outside a $10\%$ reference set and a $20\%$ test set (padded to $311{,}296$), $\Rp$ holds the first feature, $\Op$ the other two and $y$, block normalization, $12{,}000$ matched records, settings I and P, $\varepsilon \in \{1, 4\}$. Before running we required, at each size of (A), exact linkage, the ridge and the estimates inside the $95\%$ range of a plaintext simulation of the same mechanism on the same data, a noise s.d.\ within $3\%$ of its calibrated value, and every phase within $25\%$ of the cost model; the last failed, as Section~\ref{sec:cost} describes.
\begin{itemize}[noitemsep]
  \item \emph{Linkage.} In all $78$ runs $\Rp$'s match set and every position agreed with the plaintext join. No felt-wise false match occurred, and the cuckoo insertion never needed its fallback.
  \item \emph{Noise.} Pooled over the $78$ runs, the $3.04$ million decrypted slot values outside the packed entries have mean $-0.0007$ (standard error $0.0006$) and s.d.\ $0.9997$ (Figure~\ref{fig:noise_large}). The released entries have s.d.\ $1.000$ (stage~1, $3{,}450$ values, Kolmogorov--Smirnov $p = 0.83$) and $1.012$ (stage~2, $4{,}224$ values, $p = 0.41$); per cell and release the s.d.\ lies within $0.994$--$1.007$.
  \item \emph{Against simulation.} The ridge lay inside the simulated $95\%$ range in all $78$ runs (under I at $12{,}000$ matched and $\varepsilon = 1$: $11.3$k--$11.8$k against $9.9$k--$12.9$k), and the coefficients in $0.939$--$0.975$ of cases per size of (A), with $|z| \le 3.4$ against the simulated law given the run's ridge. Under I the Wald interval covered $\btheta_\lambda$ for $344$ of $354$ coefficients ($0.972$); $400$ plaintext releases per cell on the same data cover $0.941$--$0.968$.
  \item \emph{Accuracy at $12{,}000$ matched.} The register size does not change the release, so (A) repeats the accuracy of Section~\ref{sec:eval:sim} at this $n$: the population test MSE relative to least squares on the same cohort is $1.26$--$1.33$ under I and $1.11$--$1.13$ under P at $\varepsilon = 1$ (standardized, ridge on; simulated medians $1.29$ and $1.13$--$1.14$), $1.008$--$1.018$ under P block-normalized, and at most $1.006$ at $\varepsilon = 4$.
  \item \emph{Regime B.} The ridge was off in all $12$ runs, as predicted: $n_{\mathrm{off}}$ is $52.5$k at $\varepsilon = 0.5$ and $27$k at $\varepsilon = 1$, and in simulation the ridge is on in $17\%$ of releases at $60$k and $\varepsilon = 0.5$, within the switch-off factor $1.2$, and in none elsewhere. The bias-aware interval for $\bbeta$ was finite in every run, with median half-widths $0.055$ and $0.029$ at $60$k ($\varepsilon = 0.5$, $1$) and $0.034$ and $0.018$ at $95$k, and in simulation it covers the true slopes $0.941$--$0.969$ of the time. Prediction is within $1.2\%$ of least squares; the coefficient RMSE is $9.1$, $4.7$, $4.7$ and $2.6$ times that of least squares.
  \item \emph{3droad.} The ridge was off in every release, and the test MSE lay within $0.9995$--$1.0014$ of the block-normalized non-private fit in every run. Block normalization reduces $\Rp$'s single feature to its sign, so that fit predicts little better than the mean ($0.973$ against $0.994$ on standardized $y$). The Wald interval covered $23$ of $24$ coefficients.
\end{itemize}
Figure~\ref{fig:accuracy_large} extends the simulation of Section~\ref{sec:eval:sim} to $10^6$ matched records, with the encrypted runs as spot checks. From $10^5$ matched records prediction is within $0.5\%$ of least squares at every $\varepsilon \ge 0.5$ in each setting, the ridge is off, the Wald interval covers $0.945$--$0.966$, and the coefficient RMSE relative to least squares falls to $1.45$ ($\varepsilon = 0.5$) and $1.15$ ($\varepsilon = 1$) at $10^6$ under I.

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/accuracy_large.png}
  \caption{Plaintext simulation at $12$k to $10^6$ matched records ($100$ cohorts per $n$, $\delta = 10^{-9}$, $R^2 = 0.5$): population test MSE relative to least squares on the same cohort minus one (top) and coefficient RMSE relative to least squares (bottom). Hollow markers: the ridge is on in most releases. Crosses: the encrypted runs of regimes A ($12$k) and B ($60$k, $95$k).}
  \label{fig:accuracy_large}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/noise_large.png}
  \caption{Decrypted noise of every release of the $78$ large runs, standardized by the calibrated $\sigma$ (all slots outside the packed entries, real and imaginary parts), against $\mathcal N(0, 1)$; one panel per stage of the runs ($10^5$, $3 \times 10^5$, $10^6$ rows, regime B, 3droad).}
  \label{fig:noise_large}
\end{figure}

\subsection{Coefficient accuracy against intersection size}""")

# ---------------------------------------------------------------------------------------------- summary O5
old = cut(r"\item PSI parameters with explicit failure probabilities (correctness only, not charged to $\delta$) and an item length that meets them at $\bar n_O = 10^6$.")
block('L09', 'Open obligations: the item length is met', old, r"""\item PSI parameters with explicit failure probabilities (correctness only, not charged to $\delta$). The item length now meets them at $\bar n_O = 10^6$: two felts per item and a label-consistency test give $p_{\mathrm{pass}} = 2^{-67.5}$ there with $n_R^{\max} = 20{,}000$ (Appendix~\ref{app:psi}), and \textsc{Setup} checks the bound.""")

# ---------------------------------------------------------------------------------------------- status table
old = cut(r"$p_{\mathrm{cuckoo}} \le 2^{-49.1}$ at the evaluation's $n_R^{\max}$ for an insertion that finds an assignment whenever one exists \\")
block('L10', 'Status table: cuckoo and false-match slack at the large runs', old, r"""$p_{\mathrm{cuckoo}} \le 2^{-49.1}$ at $n_R^{\max} = 2{,}000$ and $\le 2^{-41.1}$ up to $100{,}000$ (balanced tables) for an insertion that finds an assignment whenever one exists; $p_{\mathrm{false}} = p_{\mathrm{pass}} \le 2^{-65}$ with two felts \\""")


def main():
    commit = sys.argv[1] if len(sys.argv) > 1 else 'HEAD'
    out = [HEADER.replace('__COMMIT__', commit)]
    for bid, desc, old, new in blocks:
        out.append(f'%R-BEGIN {bid} {desc}')
        out += ['%OLD| ' + l if l else '%OLD|' for l in old.split('\n')]
        out.append('%NEW-BEGIN')
        out.append(new)
        out.append('%NEW-END')
        out.append(f'%R-END {bid}')
        out.append('')
    os.makedirs(f'{ROOT}/paper/drafts', exist_ok=True)
    with open(f'{ROOT}/paper/drafts/large.tex', 'w') as fh:
        fh.write('\n'.join(out))
    print(f'wrote paper/drafts/large.tex ({len(blocks)} blocks)')


if __name__ == '__main__':
    main()
