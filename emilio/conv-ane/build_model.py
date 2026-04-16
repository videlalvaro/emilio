#!/usr/bin/env python3
"""Build a Game of Life CoreML model for Apple Neural Engine.

One GOL generation = one conv (neighbor count + cell state encoding)
                   + ReLU-based rule (birth/survive/die).
N generations = N-layer network, pipelined through ANE's 16 cores.

Conv kernel: [[1,1,1],[1,9,1],[1,1,1]]
  - Dead cell:  output S = neighbor_count  (0..8)
  - Alive cell: output S = neighbor_count + 9  (9..17)

Rule (all via ReLU, no branching):
  alive if S in {3, 11, 12}  (birth on 3 / survive on 2 or 3 neighbors)
  eq(S,k) = relu(1 - (relu(S-k) + relu(k-S)))   # 1 iff S==k

Usage: python3 build_model.py [grid_size] [n_generations]
"""

import sys
import torch
import torch.nn as nn
import coremltools as ct


def main():
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    n_gens = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    class GOL(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            self.conv.weight = nn.Parameter(
                torch.tensor([[[[1, 1, 1],
                                [1, 9, 1],
                                [1, 1, 1]]]], dtype=torch.float32),
                requires_grad=False,
            )

        def forward(self, x):
            for _ in range(n_gens):
                s = self.conv(x)
                # eq(s, k) = relu(1 - |s - k|), via relu decomposition of abs
                is_3 = torch.relu(1 - (torch.relu(s - 3) + torch.relu(3 - s)))
                is_11 = torch.relu(1 - (torch.relu(s - 11) + torch.relu(11 - s)))
                is_12 = torch.relu(1 - (torch.relu(s - 12) + torch.relu(12 - s)))
                x = torch.clamp(is_3 + is_11 + is_12, 0.0, 1.0)
            return x

    model = GOL()
    model.eval()

    # Trace
    example = torch.zeros(1, 1, grid_size, grid_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    # Sanity check: blinker (period-2 oscillator, always 3 alive cells)
    test = torch.zeros(1, 1, grid_size, grid_size)
    test[0, 0, 5, 4:7] = 1  # horizontal blinker
    with torch.no_grad():
        out = traced(test)
    alive = int((out > 0.5).sum().item())
    assert alive == 3, f"Blinker sanity check failed: {alive} alive (expected 3)"
    print(f"PyTorch sanity check: blinker → {alive} cells ✓")

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="grid", shape=(1, 1, grid_size, grid_size))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save("GOL.mlpackage")
    print(f"Saved GOL.mlpackage — {grid_size}×{grid_size}, {n_gens} generations, "
          f"~{n_gens * 10} ops")


if __name__ == "__main__":
    main()
