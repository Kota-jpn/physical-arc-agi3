"""ARC-style on-screen HUD and cards (black background, palette colours, chunky squares) drawn with Pillow."""
import os
import textwrap
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .grid import N, rgb

W, H = 1280, 720
BLACK, WHITE, GREY, YELLOW, GREEN, RED = (0, 0, 0), (255, 255, 255), (170, 170, 170), rgb(4), rgb(3), rgb(2)
_JA_FONTS = [
    os.getenv("PAI_FONT", ""),
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_MONO_FONTS = ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]
_cache = {}


def font(size, mono=False):
    """CJK font for text; `mono=True` gives a blocky monospace face for ASCII labels (LEVEL, ACTIONS)."""
    if (size, mono) not in _cache:
        for p in (_MONO_FONTS if mono else []) + _JA_FONTS:
            if p and os.path.exists(p):
                idx = 1 if mono and p.endswith("Menlo.ttc") else 0     # Menlo Bold
                _cache[(size, mono)] = ImageFont.truetype(p, size, index=idx)
                break
        else:
            _cache[(size, mono)] = ImageFont.load_default()
    return _cache[(size, mono)]


@dataclass
class Hud:
    level: int = 1
    n_levels: int = 3
    actions: int = 0
    lamp: bool = False
    agent: str = ""
    hypothesis: str = ""
    last: str = ""                                  # last action and its observed effect
    grid: list = field(default_factory=list)        # N x N top view (ARC integers)
    lit: list = field(default_factory=list)         # bordered cells
    note: str = ""                                  # e.g. 仮説 3→1 | 検証: place([3, 1])
    speed: str = ""


def wrap(d, text, fnt, width):
    """Wrap text (Japanese has no spaces) to a pixel width."""
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=fnt) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur]


def palette_strip(d, x, y, size=26, gap=4):
    for i in range(10):
        d.rectangle([x + i * (size + gap), y, x + i * (size + gap) + size, y + size], fill=rgb(i),
                    outline=(70, 70, 70))


def compose(rgb_frame, hud):
    img = Image.fromarray(np.asarray(rgb_frame)).convert("RGB").resize((W, H))
    d = ImageDraw.Draw(img)
    # top bar
    d.rectangle([0, 0, W, 74], fill=BLACK)
    d.text((20, 6), "PHYSICAL ARC-AGI-3", font=font(30, True), fill=WHITE)
    d.text((22, 46), "unofficial / inspired by ARC-AGI-3", font=font(15, True), fill=GREY)
    x = 560
    for i in range(1, hud.n_levels + 1):          # level squares: done green, current yellow, later grey
        c = GREEN if i < hud.level else YELLOW if i == hud.level else (70, 70, 70)
        d.rectangle([x + (i - 1) * 30, 14, x + (i - 1) * 30 + 22, 36], fill=c)
    d.text((x, 42), f"LEVEL {hud.level}/{hud.n_levels}", font=font(24, True), fill=WHITE)
    d.text((780, 14), f"ACTIONS {hud.actions}", font=font(34, True), fill=YELLOW)
    d.rectangle([1010, 14, 1040, 44], fill=GREEN if hud.lamp else rgb(5), outline=WHITE, width=2)
    d.text((1052, 14), "LAMP", font=font(18, True), fill=GREY)
    d.text((1052, 38), "ON" if hud.lamp else "off", font=font(22, True), fill=GREEN if hud.lamp else GREY)
    # observation grid (what the agent sees), right side
    if hud.grid:
        cs, g, x0, y0 = 40, 3, W - 24 - N * 43, 118
        d.rectangle([x0 - 12, y0 - 34, W - 12, y0 + N * (cs + g) + 12], fill=BLACK)
        d.text((x0, y0 - 28), "OBSERVATION", font=font(16, True), fill=GREY)
        for r in range(N):
            for c in range(N):
                x, y = x0 + c * (cs + g), y0 + r * (cs + g)
                d.rectangle([x, y, x + cs, y + cs], fill=rgb(hud.grid[r][c]), outline=(60, 60, 60))
                if [r, c] in hud.lit:
                    d.rectangle([x, y, x + cs, y + cs], outline=WHITE, width=4)
    # bottom bar: hypothesis, last action, agent
    d.rectangle([0, H - 150, W, H], fill=BLACK)
    d.text((20, H - 142), "仮説 HYPOTHESIS", font=font(16), fill=YELLOW)
    for i, ln in enumerate(wrap(d, hud.hypothesis, font(26), W - 44)[:2]):
        d.text((20, H - 118 + 34 * i), ln, font=font(26), fill=WHITE)
    d.text((20, H - 44), hud.last, font=font(17, True), fill=GREY)
    d.text((20, H - 22), f"agent: {hud.agent}", font=font(17, True), fill=(130, 200, 255))
    d.text((620, H - 24), hud.note, font=font(20), fill=YELLOW)
    if hud.speed:
        d.text((W - 150, H - 30), hud.speed, font=font(22, True), fill=YELLOW)
    return np.asarray(img)


def card(lines, strip_y=620):
    """Centred ARC-style card. lines: [(text, size, colour)]; sizes <= 22 use the mono face."""
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    y = 150
    for text, size, colour in lines:
        w = d.textlength(text, font=font(size))
        d.text(((W - w) / 2, y), text, font=font(size), fill=colour)
        y += int(size * 1.6)
    palette_strip(d, (W - 10 * 30) // 2, strip_y)
    return np.asarray(img)


def title_card(agent):
    return card([("PHYSICAL ARC-AGI-3", 72, WHITE),
                 ("(unofficial, inspired by ARC-AGI-3)", 32, YELLOW),
                 ("", 20, WHITE),
                 ("ロボットには「ルール」も「ゴール」も教えない。", 34, WHITE),
                 ("試して、推測して、できるだけ少ない行動で解く。", 34, WHITE),
                 ("", 20, WHITE),
                 (f"agent: {agent}   |   Franka Panda / Genesis (CPU)", 22, GREY)])


def level_card(n, n_levels, actions, optimal):
    return card([("LEVEL COMPLETE", 80, GREEN),
                 (f"LEVEL {n}/{n_levels}", 40, WHITE),
                 (f"ACTIONS  {actions}   (best possible: {optimal})", 40, YELLOW)])


def table_card(agent, rows):
    """End card. rows: [(level, agent actions or None, random baseline text, optimal)]."""
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    d.text((60, 50), "RESULTS: actions to solve (fewer is better)", font=font(40, True), fill=WHITE)
    d.text((60, 108), f"agent: {agent}   |   unofficial, inspired by ARC-AGI-3", font=font(22, True), fill=GREY)
    cols = [60, 220, 520, 1060]
    heads = ["LEVEL", "THIS RUN", "RANDOM BASELINE", "BEST"]
    for x, t in zip(cols, heads):
        d.text((x, 190), t, font=font(26, True), fill=YELLOW)
    d.line([60, 232, W - 60, 232], fill=GREY, width=2)
    for i, (lv, mine, rnd, opt) in enumerate(rows):
        y = 256 + i * 82
        d.rectangle([cols[0], y + 6, cols[0] + 40, y + 46], fill=rgb(5))
        d.text((cols[0] + 56, y), str(lv), font=font(46, True), fill=WHITE)
        d.text((cols[1], y), "FAILED" if mine is None else str(mine), font=font(46, True),
               fill=RED if mine is None else GREEN)
        d.text((cols[2], y + 6), rnd, font=font(26, True), fill=WHITE)
        d.text((cols[3], y), str(opt), font=font(46, True), fill=GREY)
    palette_strip(d, 60, 640)
    return np.asarray(img)
