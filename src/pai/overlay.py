"""On-screen captions (instruction, LLM plan, step progress) drawn with Pillow."""
import os
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
_FONT_CANDIDATES = [
    os.getenv("PAI_FONT", ""),
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_cache = {}


def font(size):
    if size not in _cache:
        for p in _FONT_CANDIDATES:
            if p and os.path.exists(p):
                _cache[size] = ImageFont.truetype(p, size)
                break
        else:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


@dataclass
class Hud:
    title: str = ""
    instruction: str = ""
    backend: str = ""
    steps: list = field(default_factory=list)      # [(label, status)] status: todo/doing/done/failed
    plan_tag: str = ""                              # e.g. "PLAN v1" / "RE-PLAN v2"
    message: str = ""
    message_color: tuple = (255, 255, 255)
    speed: str = ""


_STATUS = {"todo": ((200, 200, 200), "○"), "doing": ((255, 215, 0), "▶"),
           "done": ((120, 230, 140), "●"), "failed": ((255, 110, 110), "×")}


def compose(rgb, hud):
    img = Image.fromarray(np.asarray(rgb)).convert("RGBA").resize((W, H))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    # top bar: instruction
    d.rectangle([0, 0, W, 78], fill=(0, 0, 0, 170))
    d.text((24, 8), f"{hud.title}  |  instruction", font=font(18), fill=(150, 200, 255, 255))
    size = 30
    while d.textlength(hud.instruction, font=font(size)) > W - 48 and size > 14:
        size -= 1
    d.text((24, 32), hud.instruction, font=font(size), fill=(255, 255, 255, 255))
    # right panel: plan
    if hud.steps:
        x0, y0 = W - 400, 96
        h = 54 + 30 * len(hud.steps)
        d.rounded_rectangle([x0, y0, W - 16, y0 + h], 12, fill=(0, 0, 0, 170))
        d.text((x0 + 16, y0 + 10), hud.plan_tag, font=font(20), fill=(255, 180, 80, 255))
        for i, (label, st) in enumerate(hud.steps):
            col, mark = _STATUS[st]
            d.text((x0 + 16, y0 + 46 + 30 * i), f"{mark} {label}", font=font(21), fill=col + (255,))
    # bottom bar
    d.rectangle([0, H - 64, W, H], fill=(0, 0, 0, 170))
    d.text((24, H - 54), hud.message, font=font(26), fill=hud.message_color + (255,))
    foot = "  ".join(s for s in [f"planner: {hud.backend}" if hud.backend else "", hud.speed] if s)
    d.text((24, H - 22), foot, font=font(15), fill=(170, 170, 170, 255))
    return np.asarray(Image.alpha_composite(img, lay).convert("RGB"))


def card(lines, sub=None):
    """Title / end card. lines: [(text, size, color)]"""
    img = Image.new("RGB", (W, H), (16, 24, 40))
    d = ImageDraw.Draw(img)
    y = 150
    for text, size, col in lines:
        w = d.textlength(text, font=font(size))
        d.text(((W - w) / 2, y), text, font=font(size), fill=col)
        y += int(size * 1.6)
    return np.asarray(img)


def text_card(title, lines, color=(230, 230, 230), size=22):
    """Left-aligned text card (used to show the LLM request / JSON answer)."""
    img = Image.new("RGB", (W, H), (16, 24, 40))
    d = ImageDraw.Draw(img)
    d.text((48, 36), title, font=font(34), fill=(255, 180, 80))
    y = 100
    for ln in lines:
        d.text((48, y), ln, font=font(size), fill=color)
        y += int(size * 1.45)
    return np.asarray(img)
