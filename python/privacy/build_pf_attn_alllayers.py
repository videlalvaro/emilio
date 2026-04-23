"""T5b: drive PF_attn{N}_T128 build + per-layer cos check vs pf_attn_alllayers.npz.

For each layer in --layers (default 1-7):
  1. Run build_pf_attn_ane.py --layer N (skip-existing).
  2. Validate: cos(ANE pred, opf attn_out) >= 0.97 on first valid sentence.
  3. If cos < gate, STOP the loop (do not proceed to next layer).

Run with Xcode python (coremltools 9):
    /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
        python/privacy/build_pf_attn_alllayers.py --layers 1-7
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = REPO_ROOT / "emilio" / "conv-ane"
GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_attn_alllayers.npz"
BUILD_SCRIPT = REPO_ROOT / "python" / "privacy" / "build_pf_attn_ane.py"
COS_GATE = 0.97
D_MODEL = 640
T_SEQ = 128


def parse_layers(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def validate_layer(layer: int, mlpkg: Path) -> tuple[float, float]:
    import coremltools as ct
    z = np.load(GOLDEN, allow_pickle=False)
    attn_in = z[f"L{layer}_attn_in"].astype(np.float32)            # (B, T, 640)
    attn_out = z[f"L{layer}_attn_out"].astype(np.float32)          # (B, T, 640)
    # Need attention mask too; reuse from pf_golden via attn_in non-zero rows? simpler: load it.
    z0 = np.load(REPO_ROOT / "python" / "privacy" / "out" / "pf_golden.npz",
                 allow_pickle=False)
    mask = z0["attention_mask"].astype(np.int32)                   # (B, T)
    B, T, D = attn_in.shape
    m = ct.models.MLModel(str(mlpkg), compute_units=ct.ComputeUnit.ALL)
    cosines, max_abs = [], []
    for i in range(B):
        n_valid = int(mask[i].sum())
        x_in = attn_in[i].T[None, :, None, :].astype(np.float16)   # (1,640,1,128)
        pad_add = np.where(mask[i] > 0, 0.0, -1e4).astype(np.float16)[None, None, None, :]
        out = m.predict({"x_in": x_in, "pad_add": pad_add})
        y = out["x_out"].reshape(D, T).T.astype(np.float32)        # (T,640)
        a = attn_out[i, :n_valid].reshape(-1)
        b = y[:n_valid].reshape(-1)
        cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-30))
        cosines.append(cos); max_abs.append(float(np.abs(a-b).max()))
    return float(np.min(cosines)), float(np.max(max_abs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1-7", help="e.g. '1-7' or '1,3,5'")
    ap.add_argument("--force", action="store_true",
                    help="Pass --force through to per-layer builder")
    args = ap.parse_args()

    if not GOLDEN.exists():
        raise SystemExit(
            f"missing {GOLDEN}; run pf_attn_alllayers_goldens.py first")

    layers = parse_layers(args.layers)
    print(f"[t5b-driver] layers: {layers}")
    rows = []
    for n in layers:
        mlpkg = PKG_DIR / f"PF_attn{n}_T128.mlpackage"
        print(f"\n[t5b-driver] === layer {n} ===")
        t0 = time.time()
        cmd = [sys.executable, str(BUILD_SCRIPT), "--layer", str(n)]
        if args.force:
            cmd.append("--force")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[t5b-driver] L{n} BUILD FAILED rc={rc}")
            rows.append((n, "BUILD_FAIL", 0.0, 0.0, time.time() - t0))
            return 1
        print(f"[t5b-driver] L{n} build done in {time.time()-t0:.1f}s; validating...")
        cos, ma = validate_layer(n, mlpkg)
        ok = cos >= COS_GATE
        rows.append((n, "PASS" if ok else "FAIL", cos, ma, time.time() - t0))
        print(f"[t5b-driver] L{n}: cos={cos:.6f}  max|Δ|={ma:.3f}  "
              f"[{'PASS' if ok else 'FAIL'}]")
        if not ok:
            print(f"[t5b-driver] STOP: cos < {COS_GATE} gate at L{n}")
            break

    print(f"\n[t5b-driver] === SUMMARY ===")
    print(f"{'L':>2}  {'cos':>10}  {'max|d|':>8}  {'wall':>6}  status")
    all_pass = True
    for n, status, cos, ma, dt in rows:
        print(f"{n:>2}  {cos:10.6f}  {ma:8.3f}  {dt:5.1f}s  {status}")
        if status != "PASS":
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
