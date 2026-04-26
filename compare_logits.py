import numpy as np

# REAP golden
r = np.load('python/moe/out/gemma_hf_golden_logits_reap.npz', allow_pickle=False)
print('=== REAP golden ===')
print('keys:', list(r.keys()))
for k in r.keys():
    v = r[k]
    print(f'  {k}: shape={v.shape} dtype={v.dtype}')
    if 'ids' in k.lower() or 'input' in k.lower():
        print(f'    values: {v.tolist()[:20]}')

# Full golden
g = np.load('python/moe/out/gemma_golden.npz', allow_pickle=False)
print()
print('=== Full golden ===')
print('keys:', list(g.keys()))
for k in g.keys():
    v = g[k]
    print(f'  {k}: shape={v.shape} dtype={v.dtype}')
    if 'ids' in k.lower() or 'prompt' in k.lower():
        if v.ndim == 0:
            print(f'    scalar: {v.item()!r}')
        else:
            print(f'    values: {v.tolist()[:20]}')

# Compare logit norms at pos 0
r_logits = r['logits']
g_logits = g['logits_full']
print()
print('pos-0 logit stats:')
print(f'  REAP pos0: norm={np.linalg.norm(r_logits[0]):.2f} max={r_logits[0].max():.4f} min={r_logits[0].min():.4f}')
print(f'  Gold pos0: norm={np.linalg.norm(g_logits[0]):.2f} max={g_logits[0].max():.4f} min={g_logits[0].min():.4f}')
print(f'  REAP pos0 top5 idx: {np.argsort(-r_logits[0])[:5].tolist()}')
print(f'  Gold pos0 top5 idx: {np.argsort(-g_logits[0])[:5].tolist()}')
print(f'  REAP argmax(pos0)={np.argmax(r_logits[0])} val={r_logits[0][np.argmax(r_logits[0])]:.4f}')
print(f'  Gold argmax(pos0)={np.argmax(g_logits[0])} val={g_logits[0][np.argmax(g_logits[0])]:.4f}')

# Check if golden is softcapped
print()
print('softcap check:')
for label, logits in [('REAP', r_logits), ('Gold', g_logits)]:
    absmax = np.abs(logits).max()
    print(f'  {label} absmax={absmax:.4f}')
    if absmax < 35:
        print(f'    looks softcapped (all < 35)')
    else:
        print(f'    NOT softcapped (max > 35)')
