"""Generates the logo and banners (SVG), then renders PNGs with headless Chrome. No photos involved: every
figure is an abstract silhouette. Run: python3 make_branding.py"""
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# WikiTree brand colours (from the WikiTree brand palette) and Calibri
DARK, LIME, LIME2, PALE, PALE2, YELLOW, ORANGE = "#25422D", "#8FC641", "#A5D167", "#E1F0B4", "#E5EECF", "#FCB815", "#f37c26"
CREAM, PAPER, INK, GREEN, SEPIA = PALE2, "#ffffff", DARK, DARK, "#3f6347"
TONES = [LIME, LIME2, PALE, "#FFE270", "#FAD158", "#c5e08a"]       # pixelated heads: a green/yellow mosaic
FONT = "Calibri, Carlito, 'Helvetica Neue', Arial, sans-serif"


def figure(cx, cy, r, color, pixel=False, seed=0):
    """Head + shoulders. pixel=True draws the head as a coarse block grid, like the tool's blur mode."""
    rng = random.Random(seed)
    body = (f'<path d="M{cx - r * 1.9:.1f},{cy + r * 3.2:.1f} Q{cx - r * 1.9:.1f},{cy + r * 1.35:.1f} {cx:.1f},{cy + r * 1.3:.1f} '
            f'Q{cx + r * 1.9:.1f},{cy + r * 1.35:.1f} {cx + r * 1.9:.1f},{cy + r * 3.2:.1f} Z" fill="{color}" opacity=".85"/>')
    if not pixel:
        return body + f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'
    n = 4
    s = 2 * r / n
    cells = "".join(f'<rect x="{cx - r + i * s:.1f}" y="{cy - r + j * s:.1f}" width="{s + .4:.1f}" height="{s + .4:.1f}" fill="{rng.choice(TONES)}"/>'
                    for i in range(n) for j in range(n))
    return body + cells


def print_card(x, y, w, h, rot, inner):
    return (f'<g transform="translate({x},{y}) rotate({rot})">'
            f'<rect x="{-w / 2 + 6}" y="{-h / 2 + 8}" width="{w}" height="{h}" rx="8" fill="#000" opacity=".12"/>'
            f'<rect x="{-w / 2}" y="{-h / 2}" width="{w}" height="{h}" rx="8" fill="{PAPER}" stroke="#cfe0a6"/>{inner}</g>')


def brackets(cx, cy, r, color=ORANGE, w=5):
    a, L = r * 1.35, r * 0.55
    d = ""
    for sx in (-1, 1):
        for sy in (-1, 1):
            d += f'M{cx + sx * a:.1f},{cy + sy * a + -sy * L:.1f} L{cx + sx * a:.1f},{cy + sy * a:.1f} L{cx + sx * (a - L):.1f},{cy + sy * a:.1f} '
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>'


def chip(x, y, text, size=15):
    wd = len(text) * size * 0.54 + 26
    return (f'<g transform="translate({x},{y})"><rect width="{wd:.0f}" height="{size + 16}" rx="{(size + 16) / 2}" fill="{YELLOW}"/>'
            f'<text x="13" y="{size + 3}" font-family="{FONT}" font-weight="700" font-size="{size}" fill="{DARK}">{text}</text></g>')


def group_photo(w, h, named=True):
    """A 2-row group of silhouettes; every head pixelated except one named, deceased person."""
    parts, late, seed = [], [], 1
    rows = [(0.34, 0.075, 7, 0.0), (0.74, 0.1, 5, 0.0)]
    clear_idx = 2
    for ri, (yy, rr, n, _) in enumerate(rows):
        for i in range(n):
            cx = -w / 2 + w * (i + 0.5) / n + (w / n * 0.1 if ri else 0)
            cy = -h / 2 + h * yy - rr * h * 1.3
            r = rr * h
            col = [DARK, "#355a3f", "#476f4d", "#2c4d35"][(i + ri) % 4]
            is_clear = named and ri == 1 and i == clear_idx
            parts.append(figure(cx, cy, r, col, pixel=not is_clear, seed=seed))
            seed += 1
            if is_clear:
                late.append(brackets(cx, cy, r))
                late.append(chip(cx - 52, cy + r * 1.75, "deceased ✓", 15))
    return "".join(parts + late)   # label and brackets last, so no silhouette covers them


def wrap(w, h, body, bg=CREAM):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{bg}"/><stop offset="1" stop-color="{PALE}"/></linearGradient></defs>'
            f'<rect width="{w}" height="{h}" fill="url(#g)"/>{body}</svg>')


def logo(size=512):
    s = size
    inner = (f'<rect x="{-s * .30}" y="{-s * .33}" width="{s * .60}" height="{s * .50}" rx="6" fill="{PALE}"/>'
             + figure(-s * .06, -s * .15, s * .075, DARK) + brackets(-s * .06, -s * .15, s * .075)
             + figure(s * .13, -s * .10, s * .055, "#476f4d", pixel=True, seed=3)
             + f'<text x="{-s * .05}" y="{s * .26}" text-anchor="middle" font-family="{FONT}" font-weight="700" font-size="{s * .075}" fill="{INK}">PhotoTag</text>')
    card = print_card(s / 2, s / 2, s * .68, s * .80, -4, inner)
    tag = (f'<g transform="translate({s * .72},{s * .73}) rotate(8)"><rect width="{s * .18}" height="{s * .18}" rx="{s * .045}" fill="{LIME}"/>'
           f'<path d="M{s * .045},{s * .09} l{s * .036},{s * .04} l{s * .063},{-s * .08}" fill="none" stroke="{DARK}" stroke-width="{s * .03}" stroke-linecap="round" stroke-linejoin="round"/></g>')
    return wrap(s, s, card + tag)


def banner(w, h):
    k, w0, h0 = 1, 1000, 420     # drawn on a 1000x420 canvas, then scaled and centred into w x h
    scale = min(w / w0, h / h0)
    left = (f'<text x="{60 * k}" y="{146 * k}" font-family="{FONT}" font-weight="700" font-size="{72 * k}" fill="{INK}">WikiTree</text>'
            f'<text x="{60 * k}" y="{218 * k}" font-family="{FONT}" font-weight="700" font-size="{72 * k}" fill="{ORANGE}">PhotoTag</text>'
            f'<text x="{60 * k}" y="{268 * k}" font-family="{FONT}" font-size="{27 * k}" fill="{INK}">Name every face once.</text>'
            f'<text x="{60 * k}" y="{300 * k}" font-family="{FONT}" font-size="{27 * k}" fill="{SEPIA}">Only the deceased leave your computer.</text>'
            f'<text x="{60 * k}" y="{356 * k}" font-family="{FONT}" font-weight="700" font-size="{15 * k}" letter-spacing="1.5" fill="{SEPIA}">OPEN MODELS · 100% LOCAL · COMMUNITY TOOL</text>')
    cw, ch = 400 * k, 300 * k
    card = print_card(w0 - 270 * k, h0 / 2 + 4 * k, cw, ch, 2.5, f'<g transform="scale(1)">{group_photo(cw - 40 * k, ch - 40 * k)}</g>')
    ox, oy = (w - w0 * scale) / 2, (h - h0 * scale) / 2
    return wrap(w, h, f'<g transform="translate({ox:.1f},{oy:.1f}) scale({scale:.4f})">{left}{card}</g>')


for name, svg in {"logo.svg": logo(512), "banner-dev-1000x420.svg": banner(1000, 420), "social-preview-1280x640.svg": banner(1280, 640)}.items():
    (HERE / name).write_text(svg, encoding="utf-8")
    w, h = {"logo.svg": (512, 512), "banner-dev-1000x420.svg": (1000, 420), "social-preview-1280x640.svg": (1280, 640)}[name]
    html = HERE / f"_{name}.html"
    html.write_text(f'<html><body style="margin:0">{svg}</body></html>')
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--user-data-dir=/private/tmp/claude-501/chrome-brand", f"--window-size={w},{h}",
                    f"--screenshot={HERE / name.replace('.svg', '.png')}", f"file://{html.resolve()}"], capture_output=True, timeout=60)
    html.unlink()
print("done")
