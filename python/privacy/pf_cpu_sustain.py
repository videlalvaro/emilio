"""CPU-baseline sustain workload for the privacy-filter energy bench.

Loads the PyTorch reference model (bf16 → cast to fp32 on CPU), then runs a
forward over the 8 golden sentences in a loop until PF_SUSTAIN_S seconds
elapse. Emits BEGIN_AT/END_AT/ITERS lines compatible with ane_energy_parse.py.

Invoke under powermetrics capture (see scripts/pf_energy_probe.sh). Do NOT
run by hand for benchmarking — the loader cost is significant; the BEGIN_AT
marker is emitted AFTER the first warm forward so the energy window is
steady-state only.

Env:
    PF_SUSTAIN_S   wall-clock seconds to sustain (default 30)
    PYTHONPATH     must include python/privacy/_vendor_src/opf_src
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
REF_GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_golden.npz"


def main() -> int:
    sustain_s = float(os.environ.get("PF_SUSTAIN_S", "30"))

    if not REF_GOLDEN.exists():
        raise SystemExit(f"missing {REF_GOLDEN}")

    z = np.load(REF_GOLDEN, allow_pickle=False)
    input_ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    attn = torch.from_numpy(z["attention_mask"].astype(np.int64))
    n_sent = int(input_ids.shape[0])

    torch.set_num_threads(int(os.environ.get("PF_TORCH_THREADS", "8")))

    from opf._model.model import Transformer  # type: ignore

    print(f"[cpu] loading model (CPU, fp32) ...", flush=True)
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()

    print(f"[cpu] warmup forward ...", flush=True)
    with torch.inference_mode():
        _ = model(input_ids[:1], attention_mask=attn[:1])

    # Steady-state window starts NOW.
    begin = time.time()
    print(f"BEGIN_AT {begin:.3f}", flush=True)
    deadline = begin + sustain_s
    iters = 0
    with torch.inference_mode():
        while time.time() < deadline:
            b = iters % n_sent
            _ = model(input_ids[b:b + 1], attention_mask=attn[b:b + 1])
            iters += 1
    end = time.time()
    print(f"END_AT   {end:.3f}", flush=True)
    print(f"ITERS    {iters}", flush=True)
    if end > begin:
        print(f"throughput {iters / (end - begin):.2f} sentences/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
