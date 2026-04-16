#!/usr/bin/env python3
"""Generate animated GIFs for the Conv-ANE blog post."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = "gifs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ──────────────────────────────────────────────────────
BG       = (24, 24, 32)       # dark background
ALIVE    = (45, 125, 70)      # green alive cells
DEAD     = (40, 40, 50)       # dark dead cells
GRID_CLR = (50, 50, 60)       # grid lines
TEXT_CLR  = (220, 220, 230)
ACCENT   = (51, 102, 204)     # blue accent
ACCENT2  = (204, 51, 51)      # red accent
ANE_CLR  = (255, 165, 0)      # orange for ANE


def try_load_font(size):
    """Try to load a monospace font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_SM = try_load_font(14)
FONT_MD = try_load_font(18)
FONT_LG = try_load_font(28)
FONT_XL = try_load_font(36)


# ═══════════════════════════════════════════════════════════════════
# GOL simulation (pure NumPy, matching the paper's convolution encoding)
# ═══════════════════════════════════════════════════════════════════

def gol_step(grid):
    """One GOL generation via convolution (paper's encoding)."""
    padded = np.pad(grid, 1, mode='constant')
    # Neighbor count via explicit shifts (no scipy needed)
    n = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
         padded[1:-1, :-2] +                     padded[1:-1, 2:] +
         padded[2:, :-2]   + padded[2:, 1:-1]   + padded[2:, 2:])
    # Conway's rules
    return ((n == 3) | ((grid == 1) & (n == 2))).astype(np.uint8)


def gosper_glider_gun():
    """Returns a grid with a Gosper glider gun at top-left."""
    grid = np.zeros((64, 80), dtype=np.uint8)
    # Standard Gosper glider gun pattern
    gun = [
        (5,1),(5,2),(6,1),(6,2),                          # left block
        (3,13),(3,14),(4,12),(4,16),(5,11),(5,17),
        (6,11),(6,15),(6,17),(6,18),(7,11),(7,17),
        (8,12),(8,16),(9,13),(9,14),                      # left part
        (1,25),(2,23),(2,25),(3,21),(3,22),(4,21),(4,22),
        (5,21),(5,22),(6,23),(6,25),(7,25),               # right part
        (3,35),(3,36),(4,35),(4,36),                       # right block
    ]
    for r, c in gun:
        grid[r, c] = 1
    return grid


def render_gol_frame(grid, gen, cell_size=8, label=True):
    """Render a GOL grid to a PIL Image."""
    h, w = grid.shape
    img_w = w * cell_size + 1
    img_h = h * cell_size + 1 + (40 if label else 0)
    img = Image.new('RGB', (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)

    y_off = 40 if label else 0

    for r in range(h):
        for c in range(w):
            x0 = c * cell_size
            y0 = r * cell_size + y_off
            color = ALIVE if grid[r, c] else DEAD
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                          fill=color)

    if label:
        draw.text((8, 8), f"Game of Life · Generation {gen}",
                  fill=TEXT_CLR, font=FONT_MD)
        draw.text((img_w - 200, 8), "ANE Conv2d(3×3)",
                  fill=ANE_CLR, font=FONT_SM)

    return img


def make_gol_gif():
    """GIF 1: GOL evolution with Gosper glider gun."""
    print("GIF 1: GOL evolution...")
    grid = gosper_glider_gun()
    frames = []
    n_gens = 300  # 10 seconds at 30fps

    for gen in range(n_gens):
        if gen % 30 == 0:
            print(f"  gen {gen}/{n_gens}")
        frames.append(render_gol_frame(grid, gen))
        grid = gol_step(grid)

    path = os.path.join(OUT_DIR, "gol_evolution.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=33, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 2: Token-by-token transformer generation
# ═══════════════════════════════════════════════════════════════════

def render_token_frame(prompt, tokens_so_far, tok_idx, total_tokens,
                       ms_per_tok=5.67, width=720, height=400):
    """Render one frame of token-by-token generation."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((20, 15), "Qwen2.5-0.5B on Apple Neural Engine",
              fill=TEXT_CLR, font=FONT_LG)
    draw.text((20, 50), "Conv2d(1×1) · Stateful KV · 171 tok/s",
              fill=ANE_CLR, font=FONT_SM)

    # Separator
    draw.line([(20, 75), (width - 20, 75)], fill=GRID_CLR, width=1)

    # Prompt (dimmed)
    draw.text((20, 90), "Prompt:", fill=(120, 120, 130), font=FONT_SM)
    draw.text((20, 110), prompt, fill=(150, 150, 160), font=FONT_MD)

    # Generated text
    draw.text((20, 150), "Output:", fill=(120, 120, 130), font=FONT_SM)

    # Word-wrap the generated text
    text = tokens_so_far
    y = 175
    line = ""
    max_chars = 65
    for ch in text:
        line += ch
        if len(line) >= max_chars and ch == ' ':
            draw.text((20, y), line.rstrip(), fill=TEXT_CLR, font=FONT_MD)
            y += 24
            line = ""
    if line:
        # Draw text so far + blinking cursor
        draw.text((20, y), line, fill=TEXT_CLR, font=FONT_MD)
        # Cursor
        bbox = FONT_MD.getbbox(line)
        cursor_x = 20 + bbox[2]
        if tok_idx % 4 < 2:  # blink
            draw.rectangle([cursor_x + 2, y, cursor_x + 4, y + 20],
                          fill=ACCENT)

    # Stats bar at bottom
    draw.line([(20, height - 55), (width - 20, height - 55)],
              fill=GRID_CLR, width=1)

    # Progress bar
    bar_x, bar_y, bar_w = 20, height - 45, width - 40
    bar_h = 8
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                   fill=DEAD)
    progress = min(tok_idx / max(total_tokens, 1), 1.0)
    draw.rectangle([bar_x, bar_y,
                    bar_x + int(bar_w * progress), bar_y + bar_h],
                   fill=ACCENT)

    # Stats text
    elapsed = tok_idx * ms_per_tok
    draw.text((20, height - 30),
              f"Token {tok_idx}/{total_tokens}  ·  {elapsed:.0f}ms  ·  "
              f"5.67ms/tok  ·  171 tok/s",
              fill=(160, 160, 170), font=FONT_SM)

    return img


def make_token_gif():
    """GIF 2: Token-by-token transformer generation."""
    print("GIF 2: Token-by-token generation...")
    prompt = "What is the Game of Life?"
    # Real output from the model
    output = ("The Game of Life is a cellular automaton devised by the "
              "British mathematician John Conway. It is a model of a "
              "simplified computer simulation of a physical system, in "
              "which each cell in a grid can be in one of a limited "
              "number of states, called \"live\" or \"dead\".")

    # Split into tokens (approximate — split on spaces and punctuation)
    tokens = []
    current = ""
    for ch in output:
        current += ch
        if ch in ' .,;:!?"\'()' or len(current) > 6:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)

    frames = []
    text_so_far = ""

    # Prefill frames (show prompt being processed)
    for i in range(15):
        frames.append(render_token_frame(prompt, "", 0, len(tokens)))

    # Token generation frames (1 token per frame, with some
    # repetition for readability — hold each token for 3 frames)
    for idx, tok in enumerate(tokens):
        text_so_far += tok
        for _ in range(3):  # hold each token 3 frames
            frames.append(render_token_frame(
                prompt, text_so_far, idx + 1, len(tokens)))

    # Hold final frame
    for _ in range(30):
        frames.append(render_token_frame(
            prompt, text_so_far, len(tokens), len(tokens)))

    path = os.path.join(OUT_DIR, "token_generation.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=33, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 3: Split-screen — GOL + Transformer on same hardware
# ═══════════════════════════════════════════════════════════════════

def render_split_frame(gol_grid, gen, text_so_far, tok_idx, total_tokens,
                       width=960, height=500):
    """Render split-screen: GOL left, transformer right, ANE center."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    mid = width // 2

    # ── Title ──
    draw.text((width // 2 - 180, 10),
              "One Primitive: Convolution on ANE",
              fill=TEXT_CLR, font=FONT_LG)

    # ── Central ANE badge ──
    badge_x = mid - 55
    badge_y = 50
    draw.rounded_rectangle([badge_x, badge_y, badge_x + 110, badge_y + 35],
                           radius=5, fill=(60, 40, 10), outline=ANE_CLR)
    draw.text((badge_x + 8, badge_y + 8), "ANE 16-core",
              fill=ANE_CLR, font=FONT_SM)

    # ── Left: GOL ──
    draw.text((30, 55), "Conv2d(3×3)", fill=ACCENT, font=FONT_SM)

    # ── Right label (placed far enough right to avoid badge) ──
    draw.text((mid + 80, 55), "Conv2d(1×1)", fill=ACCENT, font=FONT_SM)

    gol_y_off = 95
    cell_size = 6
    gh, gw = gol_grid.shape
    # Only render the interesting top-left portion
    render_h = min(gh, 55)
    render_w = min(gw, 70)
    for r in range(render_h):
        for c in range(render_w):
            x0 = 15 + c * cell_size
            y0 = gol_y_off + r * cell_size
            color = ALIVE if gol_grid[r, c] else DEAD
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                          fill=color)

    draw.text((15, height - 35), f"Gen {gen} · 2.3B cells/sec",
              fill=(140, 140, 150), font=FONT_SM)

    # ── Divider ──
    draw.line([(mid, 90), (mid, height - 10)], fill=GRID_CLR, width=1)

    # ── Right: Transformer ──
    tx = mid + 20
    draw.text((tx, 95), "Q: What is the Game of Life?",
              fill=(130, 130, 140), font=FONT_SM)

    # Generated text with word wrap
    y = 125
    line = ""
    max_chars = 45
    for ch in text_so_far:
        line += ch
        if len(line) >= max_chars and ch == ' ':
            draw.text((tx, y), line.rstrip(), fill=TEXT_CLR, font=FONT_SM)
            y += 20
            line = ""
    if line:
        draw.text((tx, y), line, fill=TEXT_CLR, font=FONT_SM)
        # cursor
        bbox = FONT_SM.getbbox(line)
        cx = tx + bbox[2]
        if tok_idx % 4 < 2:
            draw.rectangle([cx + 2, y, cx + 4, y + 16], fill=ACCENT)

    draw.text((tx, height - 35),
              f"Token {tok_idx}/{total_tokens} · 171 tok/s · 5.67ms/tok",
              fill=(140, 140, 150), font=FONT_SM)

    return img


def make_split_gif():
    """GIF 3: Split-screen GOL + Transformer on same ANE."""
    print("GIF 3: Split-screen...")
    gol_grid = gosper_glider_gun()

    output = ("The Game of Life is a cellular automaton devised by the "
              "British mathematician John Conway. It is a model of a "
              "simplified computer simulation of a physical system, in "
              "which each cell in a grid can be in one of a limited "
              "number of states.")
    tokens = []
    current = ""
    for ch in output:
        current += ch
        if ch in ' .,;:!?"\'()' or len(current) > 6:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)

    frames = []
    text_so_far = ""
    tok_idx = 0

    # Total frames: ~10 seconds at 30fps = 300 frames
    # GOL evolves every frame; tokens appear every ~4 frames
    for frame in range(300):
        if frame % 30 == 0:
            print(f"  frame {frame}/300")

        # Advance token every 4 frames (after initial 20-frame pause)
        if frame > 20 and frame % 4 == 0 and tok_idx < len(tokens):
            text_so_far += tokens[tok_idx]
            tok_idx += 1

        frames.append(render_split_frame(
            gol_grid, frame, text_so_far, tok_idx, len(tokens)))

        # GOL advances every frame
        gol_grid = gol_step(gol_grid)

    path = os.path.join(OUT_DIR, "split_screen.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=33, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
# GIF 4: Data flow through 24 transformer layers
# ═══════════════════════════════════════════════════════════════════

def render_dataflow_frame(active_layer, active_op, token_text,
                          width=800, height=500):
    """Render the data flow diagram showing which layer is active."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.text((20, 10), "Token data flow: 24 layers × 7 ops = 168 convolutions",
              fill=TEXT_CLR, font=FONT_MD)

    # Current token
    draw.text((20, 40), f'Processing: "{token_text}"',
              fill=(160, 160, 170), font=FONT_SM)

    # ── Layer grid: 24 rows × 7 columns ──
    ops = ["RMSNorm", "QKV", "Attn", "Out", "RMSNorm", "FFN↑", "FFN↓"]
    grid_x, grid_y = 30, 75
    cell_w, cell_h = 100, 15
    gap = 2

    # Column headers
    for col, op in enumerate(ops):
        x = grid_x + col * (cell_w + gap)
        draw.text((x + 5, grid_y - 15), op,
                  fill=(120, 120, 130), font=FONT_SM)

    for layer in range(24):
        y = grid_y + layer * (cell_h + gap)
        # Layer label
        draw.text((grid_x - 25, y + 1), f"{layer:2d}",
                  fill=(100, 100, 110), font=FONT_SM)

        for col in range(7):
            x = grid_x + col * (cell_w + gap)

            if layer < active_layer or (layer == active_layer and col <= active_op):
                # Completed
                color = (30, 70, 45)  # dim green
            elif layer == active_layer and col == active_op + 1:
                # Currently active
                color = ACCENT
            else:
                # Pending
                color = DEAD

            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=color)

    # Active indicator
    if active_layer < 24:
        cur_op = min(active_op + 1, 6)
        ax = grid_x + cur_op * (cell_w + gap)
        ay = grid_y + active_layer * (cell_h + gap)
        draw.rectangle([ax - 1, ay - 1, ax + cell_w + 1, ay + cell_h + 1],
                       outline=ANE_CLR, width=2)

    # LM Head
    lm_y = grid_y + 24 * (cell_h + gap) + 5
    lm_done = active_layer >= 24
    lm_color = (30, 70, 45) if lm_done else DEAD
    draw.rectangle([grid_x, lm_y, grid_x + 7 * (cell_w + gap) - gap, lm_y + cell_h + 4],
                   fill=lm_color)
    draw.text((grid_x + 250, lm_y + 2), "LM Head → argmax → next token",
              fill=TEXT_CLR if lm_done else (80, 80, 90), font=FONT_SM)

    # Stats
    total_ops = 24 * 7 + 1  # 168 + LM head
    done_ops = active_layer * 7 + (active_op + 1) + (1 if active_layer >= 24 else 0)
    draw.text((20, height - 30),
              f"Op {min(done_ops, total_ops)}/{total_ops}  ·  "
              f"Layer {min(active_layer, 23)}/23  ·  5.67ms total per token",
              fill=(140, 140, 150), font=FONT_SM)

    return img


def make_dataflow_gif():
    """GIF 4: Data flow through 24 layers."""
    print("GIF 4: Data flow diagram...")
    frames = []
    token_text = "Life"

    # Walk through all 24 layers × 7 ops + LM head
    # Hold each op for 2 frames
    for layer in range(24):
        for op in range(7):
            for _ in range(2):
                frames.append(render_dataflow_frame(layer, op, token_text))

    # LM head
    for _ in range(10):
        frames.append(render_dataflow_frame(24, 0, token_text))

    # Hold final
    for _ in range(30):
        frames.append(render_dataflow_frame(25, 0, token_text))

    path = os.path.join(OUT_DIR, "dataflow.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=33, loop=0, optimize=True)
    print(f"  → {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_gol_gif()
    make_token_gif()
    make_split_gif()
    make_dataflow_gif()
    print("\nDone! All GIFs in:", OUT_DIR)
