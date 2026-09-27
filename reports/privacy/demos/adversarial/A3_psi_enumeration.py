"""
A3 — PSI membership-oracle abuse: enumeration counts (LOCAL ARITHMETIC ONLY).

R's PSI query set is unrestricted: R chooses the identifiers it puts in the cuckoo
tables (party_r.py:66-86). O evaluates its membership polynomial at every query and
returns, per match, a zero-test AND O's canonical row index as the label
(party_o.py:94-96; recovered at party_r.py:92-94). So PSI is a membership+index
oracle over ARBITRARY identifiers R chooses to submit -- fabricated or enumerated,
not restricted to any ethics-approved cohort.

This script just does the batching arithmetic; no crypto is run.
"""
import math

ring_dim = 16384                       # phase0_bfv ring_dim / num_bins (calibration default)
NB = ring_dim                          # one identifier per SIMD slot
cuckoo_cap = int(0.9 * ring_dim)       # party_r cuckoo_capacity (calibrate line 283)
print(f"BFV ring_dim = num_bins = {ring_dim}")
print(f"cuckoo capacity per batch = {cuckoo_cap}  (IDs testable per batch)\n")

# A single protocol connection lets R stream n_batches batches (party_r loops over all
# tables in one session), so R can test n_batches * cuckoo_cap IDs per connection.
for space, label in [(10**9, "9-digit SIN space (all SINs)"),
                     (10**7, "one province's plausible SIN block"),
                     (1, "one targeted individual (known SIN)")]:
    batches = math.ceil(space / cuckoo_cap)
    print(f"{label:35s}: {space:>13,} IDs -> {batches:>10,} batches "
          f"(~{batches:,} round-trips of BFV ciphertexts)")

print("""
Consequences:
 * Targeted membership: 'is person X (SIN known) in O's administrative dataset?'
   costs ONE slot. For an agency that is benefit/tax/health enrolment, mere
   membership is sensitive disclosure -- and O only ever learns n=|I|, so it cannot
   see that R is probing.
 * Each hit also returns O's row index (the label). Feeding that index into the A1
   one-hot extraction reads that identified person's (x_O, y) exactly. PSI supplies
   the WHO (real identity <-> O row index); A1 supplies the WHAT (their record).
 * Full-space enumeration (~68k batches) is heavy but offline-feasible for a
   determined adversary; targeted enumeration of a known cohort is trivial.
 * No ethics-approved-ID gate exists: nothing binds R's query set to any authorised
   population. Under the semi-honest model R is assumed not to do this; nothing
   enforces it.
""")
