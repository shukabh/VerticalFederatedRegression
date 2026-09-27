# VerticalFederatedRegression

This repository holds a two-party, privacy-preserving vertical federated OLS protocol, for
"Scenario B" of the working draft *Privacy-Preserving Vertical Federated Linear Regression
when the Organization Holds the Response* (31 Aug 2026).

The two parties are:
- a **researcher R**, who holds features `X_R` and identifiers;
- an **organization O**, which holds features `X_O`, the response `y` and identifiers.

The protocol runs in four steps:
1. They link records with PSI (BFV).
2. R sends an encrypted selection vector and its aligned features (CKKS).
3. O computes the O-dependent Gram and moment blocks under encryption and adds Gaussian DP noise.
4. R decrypts, adds its own exact block `A`, solves, and applies a bias correction that
   costs no privacy budget.

The threat model is **honest-but-curious**.

## Layout

| Path | What it is |
|---|---|
| `scenario_b/party_r.py`, `party_o.py`, `run_protocol.py`, `calibrate_hyperparameters.py`, `generate_vfl_data.py` | Protocol code, unmodified. |
| `scenario_b/psi_common.py`, `phase1_common.py`, `he_backend.py` | Imported modules: PSI primitives, the Phase-1 maths (noise, ridge gate, bias correction), and the plaintext/OpenFHE backends. |
| `scenario_b/prepare_data.py` | Adapter: runs the generator and writes the `y_O.csv` the parties read. |
| `experiments/accuracy_study.py` | Accuracy comparison: true β, OLS on the true matches, plaintext linkage + OLS, and the private protocol across ε and n. |
| `tests/test_scenario_b.py` | Tests of the modules: bias operator vs exact second moment, noise structure, plaintext and OpenFHE PSI round trips, ridge-gate branches, framing. |
| `reports/code_review.md` | The implementation checked against the draft. |
| `reports/privacy/README.md` | Consolidated privacy risks under honest-but-curious; detailed reports and demos are alongside it. |
| `reports/accuracy/accuracy_report.md` | Accuracy results, figures and CSV/JSON output. |
| `reports/checks/` | Numerical checks of Theorem 1, the adaptive ridge, and the replace-one sensitivity. |

## Run it

```bash
pip install numpy pandas scipy matplotlib pytest
cd scenario_b
python prepare_data.py --n_R 500 --d_R 5 --n_O 4000 --d_O 5 --n_intersect 400 --seed 42 --out_dir vfl_B
python calibrate_hyperparameters.py --data_dir vfl_B --eps 1 --delta 1e-5 --B_R 2.2361 --B_O 2.2361 --B_y 5
python run_protocol.py --data_dir vfl_B --backend plaintext      # launches R and O over loopback
```

About the `openfhe` backend: the `openfhe` wheel imports on Python 3.12, but the
wheel pip installs for 3.11 does not. Run the same commands with a 3.12 interpreter and
`--backend openfhe`.

To run the tests and rebuild the accuracy report:

```bash
python -m pytest -q tests/
python experiments/accuracy_study.py --quick         # smoke test; see --help for the full grid
```
