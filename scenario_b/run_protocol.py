"""Launch R and O as two processes over loopback, then verify against the OLS oracle."""
import argparse, json, os, subprocess, sys, time
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",default="vfl_B")
    ap.add_argument("--backend",default="plaintext",choices=["openfhe","plaintext"])
    ap.add_argument("--port",type=int,default=65432)
    a=ap.parse_args()
    env=dict(os.environ)
    r=subprocess.Popen([sys.executable,"party_r.py","--data_dir",a.data_dir,
                        "--backend",a.backend,"--port",str(a.port)],env=env)
    time.sleep(2.0)
    o=subprocess.Popen([sys.executable,"party_o.py","--data_dir",a.data_dir,
                        "--backend",a.backend,"--port",str(a.port)],env=env)
    rc_o=o.wait(); rc_r=r.wait()
    if rc_r or rc_o:
        print(f"FAILED (R rc={rc_r}, O rc={rc_o})"); return 1
    # ---- verify ----
    d=a.data_dir
    sig=pd.read_csv(os.path.join(d,"sigma.csv"))
    X_R=pd.read_csv(os.path.join(d,"X_R.csv")).to_numpy()[:,1:]
    X_O=pd.read_csv(os.path.join(d,"X_O.csv")).to_numpy()[:,1:]
    y=pd.read_csv(os.path.join(d,"y.csv"))["y"].to_numpy()
    Z=np.hstack([X_R[sig["researcher_idx"]],X_O[sig["org_idx"]]])
    beta_ols=np.linalg.solve(Z.T@Z,Z.T@y)
    bt=pd.read_csv(os.path.join(d,"beta_true.csv"))["beta"].to_numpy()
    out=pd.read_csv(os.path.join(d,"beta_private.csv"))
    summ=json.load(open(os.path.join(d,"run_summary.json")))
    print("\n=== verification ===")
    print(f"  matched by PSI : {summ['n_matched']} / ground truth {len(sig)} "
          f"-> {'OK' if summ['n_matched']==len(sig) else 'MISMATCH'}")
    print(f"  backend={summ['backend']}  sigma={summ['sigma']:.4g}  "
          f"Psi={summ['Psi']} lambda={summ['lambda']:.4g} rho_hat={summ['rho_hat']:.2f}")
    po=os.path.join(d,"beta_original_scale.csv")
    if os.path.exists(po):
        og=pd.read_csv(po); sl=og[og["party"]!="intercept"]["beta_original_scale"].to_numpy()
        icpt=float(og[og["party"]=="intercept"]["beta_original_scale"].iloc[0])
        print(f"  [standardized run] intercept={icpt:.4f}; comparing on ORIGINAL scale")
        print(f"  ||bias-corrected(orig) - beta_OLS|| = {np.linalg.norm(sl-beta_ols):.4f}   "
              f"||. - beta_true|| = {np.linalg.norm(sl-bt):.4f}")
    else:
        for nm,col in [("private ridge","beta_ridge"),("bias-corrected","beta_bias_corrected")]:
            v=out[col].to_numpy()
            print(f"  ||{nm:14s} - beta_OLS|| = {np.linalg.norm(v-beta_ols):.4f}   "
                  f"||. - beta_true|| = {np.linalg.norm(v-bt):.4f}")
    print(f"  ||beta_OLS - beta_true|| = {np.linalg.norm(beta_ols-bt):.4f}  (sampling only)")
    return 0

if __name__=="__main__": raise SystemExit(main())
