"""On-screen HUD and explanation cards (1920x1080, ARC palette on black) drawn with Pillow.

The HUD is written for a first-time viewer: Japanese labels, the action in words, what changed, and the
number of hypotheses still alive. The camera image fills the left 1440x810 area under the top bar.
"""
import json
import os
import re
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .game import CUBES
from .grid import N, rgb

W, H = 1920, 1080
TOP, BOTTOM, PANEL = 90, 900, 1440            # top bar height, bottom bar top, right panel left edge
CAM_RES = (PANEL, BOTTOM - TOP)               # camera image size (16:9)
BLACK, WHITE, GREY, DARK = (0, 0, 0), (255, 255, 255), (170, 170, 170), (60, 60, 60)
YELLOW, GREEN, RED, CYAN = rgb(4), rgb(3), rgb(2), (130, 200, 255)
_BOLD = [os.getenv("PAI_FONT", ""), "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]
_REGULAR = ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"] + _BOLD
_MONO = ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]
_cache = {}
COLOUR_JA = {"red": "赤", "blue": "青", "green": "緑", "yellow": "黄"}
_JA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "video", "hypo_ja.json")
HYPO_JA = json.load(open(_JA_PATH)) if os.path.exists(_JA_PATH) else {}   # hypothesis descriptions -> Japanese (video only)


def font(size, kind="bold"):
    """kind: 'bold' / 'regular' (Japanese text) or 'mono' (blocky ASCII labels)."""
    if (size, kind) not in _cache:
        paths = {"bold": _BOLD, "regular": _REGULAR, "mono": _MONO + _BOLD}[kind]
        for p in paths:
            if p and os.path.exists(p):
                idx = 1 if p.endswith("Menlo.ttc") else 0     # Menlo Bold
                _cache[(size, kind)] = ImageFont.truetype(p, size, index=idx)
                break
        else:
            _cache[(size, kind)] = ImageFont.load_default()
    return _cache[(size, kind)]


# ---- words for the viewer ----------------------------------------------------------------------
def ja_action(action):
    if action["name"] == "pick":
        return f"{COLOUR_JA.get(action['object_id'], action['object_id'])}のキューブをつかむ"
    r, c = action["cell"]
    return f"マス ({r},{c}) に置く"


def ja_effect(effect):
    """game.describe_effect() text -> short Japanese."""
    if "grasp failed" in effect:
        return "つかみ損ねた"
    parts = []
    if "floor tiles changed colour" in effect:
        parts.append("灰色のタイルに色が現れた!")
    if "newly bordered/lit" in effect:
        parts.append("タイルが反応した(白い枠)")
    if "no longer lit" in effect:
        parts.append("白い枠が消えた")
    if "lamp turned ON" in effect:
        parts.append("ランプ点灯 = クリア!")
    return "、".join(parts) or "変化なし"


def ja_said(text):
    """The model sometimes misspells maroon (マルーン) as マラソン / マーロン; show it as 栗色 (the colour name used on screen)."""
    return re.sub(r"マラソン|マーロン", "栗色", text or "")


def ja_note(note):
    """'仮説 3→1 | 検証: ...' -> ('3 → 1', purpose)."""
    m = re.search(r"仮説 (\d+)→(\d+)", note or "")
    if not m:
        return "", ""
    purpose = ("仮説を見分けるための実験" if "検証" in note else "答えが絞れたので解きに行く" if "実行" in note
               else "持ち上げて重さを測る実験" if "重さを測る" in note else "")
    return f"{m[1]} → {m[2]}", purpose


@dataclass
class Hud:
    level: int = 1
    n_levels: int = 3
    actions: int = 0
    optimal: int = 6
    lamp: bool = False
    agent: str = ""
    hypothesis: str = ""
    action: str = ""                                # current action in words
    effect: str = ""                                # what it changed, in words
    grid: list = field(default_factory=list)        # N x N top view (ARC integers)
    lit: list = field(default_factory=list)         # bordered cells
    target: list = None                             # cell the current action touches
    note: str = ""                                  # agent note, e.g. 仮説 3→1 | 検証: place([3, 1])
    callouts: list = field(default_factory=list)    # [(x, y, text, colour)] in camera-image pixels
    speed: str = ""
    felt: dict = field(default_factory=dict)        # cube -> weight [kg] the arm felt in this level
    failed: set = field(default_factory=set)        # levels not solved within the cap
    pool: list = field(default_factory=list)        # replay only: descriptions of the hypotheses still alive
    gone: list = field(default_factory=list)        # replay only: hypotheses discarded by the last observation
    banner: bool = False                            # show the "discarded" banner over the camera image


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


def palette_strip(d, x, y, size=30, gap=6):
    for i in range(10):
        d.rectangle([x + i * (size + gap), y, x + i * (size + gap) + size, y + size], fill=rgb(i), outline=(70, 70, 70))


def draw_grid(d, grid, lit, target, x0, y0, cs=62, g=5):
    for r in range(N):
        for c in range(N):
            x, y = x0 + c * (cs + g), y0 + r * (cs + g)
            d.rectangle([x, y, x + cs, y + cs], fill=rgb(grid[r][c]), outline=DARK)
            if [r, c] in lit:
                d.rectangle([x, y, x + cs, y + cs], outline=WHITE, width=6)
            if target == [r, c]:
                d.rectangle([x - 4, y - 4, x + cs + 4, y + cs + 4], outline=YELLOW, width=5)


def compose(rgb_frame, hud):
    img = Image.new("RGB", (W, H), BLACK)
    img.paste(Image.fromarray(np.asarray(rgb_frame)).convert("RGB").resize(CAM_RES), (0, TOP))
    d = ImageDraw.Draw(img)
    for x, y, text, colour in hud.callouts:        # point at board changes hidden under a cube
        y += TOP
        d.ellipse([x - 46, y - 30, x + 46, y + 30], outline=YELLOW, width=5)
        f = font(32)
        tw = d.textlength(text, font=f)
        d.line([x, y - 30, x, y - 80], fill=YELLOW, width=4)
        d.rectangle([x - tw / 2 - 14, y - 132, x + tw / 2 + 14, y - 80], fill=BLACK, outline=YELLOW, width=3)
        d.text((x - tw / 2, y - 126), text, font=f, fill=colour)
    # top bar: title, level, action count, lamp
    d.text((28, 10), "PHYSICAL ARC-AGI-3", font=font(40, "mono"), fill=WHITE)
    d.text((30, 58), "非公式・ARC-AGI-3 に着想を得た個人作品", font=font(20, "regular"), fill=GREY)
    x = 640
    for i in range(1, hud.n_levels + 1):          # level squares: done green, current yellow, later grey
        c = RED if i in hud.failed else GREEN if i < hud.level else YELLOW if i == hud.level else DARK
        d.rectangle([x + (i - 1) * 34, 14, x + (i - 1) * 34 + 26, 40], fill=c)
    d.text((x, 46), f"LEVEL {hud.level} / {hud.n_levels}", font=font(32, "mono"), fill=WHITE)
    d.text((960, 8), "行動回数", font=font(22), fill=GREY)
    d.text((960, 32), f"{hud.actions}", font=font(48, "mono"), fill=YELLOW)
    d.text((1060, 52), f"(最短 {hud.optimal})", font=font(24, "regular"), fill=GREY)
    d.ellipse([PANEL + 30, 18, PANEL + 84, 72], fill=GREEN if hud.lamp else rgb(5), outline=WHITE, width=3)
    d.text((PANEL + 100, 10), "ランプ (ゴール)", font=font(22), fill=GREY)
    d.text((PANEL + 100, 38), "点灯 = クリア!" if hud.lamp else "消灯 = まだ", font=font(32),
           fill=GREEN if hud.lamp else WHITE)
    # right panel: what the AI sees, legend, felt weights, the hypotheses the AI wrote as code
    x0 = PANEL + 38
    d.text((x0, TOP + 14), "AI が受け取る盤面", font=font(28), fill=WHITE)
    d.text((x0, TOP + 50), "(色を数字 0〜9 にした 6×6 の表)", font=font(20, "regular"), fill=GREY)
    if hud.grid:
        draw_grid(d, hud.grid, hud.lit, hud.target, x0 + 60, TOP + 84, cs=46, g=4)
    y = TOP + 394
    d.rectangle([x0, y + 4, x0 + 26, y + 30], outline=WHITE, width=5)
    d.text((x0 + 38, y + 2), "反応したタイル", font=font(22, "regular"), fill=WHITE)
    d.rectangle([x0 + 220, y + 4, x0 + 246, y + 30], outline=YELLOW, width=5)
    d.text((x0 + 258, y + 2), "いま触るマス", font=font(22, "regular"), fill=WHITE)
    if hud.felt:
        d.text((x0, y + 42), "重さ kg:", font=font(22, "regular"), fill=GREY)
        x = x0 + 92
        for cube, kg in hud.felt.items():
            t = f"{kg:.2f}"
            d.rectangle([x, y + 46, x + 20, y + 66], fill=rgb(CUBES[cube]))
            d.text((x + 26, y + 42), t, font=font(22, "regular"), fill=WHITE)
            x += 26 + d.textlength(t, font=font(22, "regular")) + 12
    y = TOP + 482
    if hud.pool or hud.gone:
        d.line([x0, y, W - 30, y], fill=DARK, width=2)
        d.text((x0, y + 10), "AI がコードで書いた仮説", font=font(28), fill=YELLOW)
        d.text((x0, y + 48), "ランプが点く条件。観察と合わなければ捨てる", font=font(19, "regular"), fill=GREY)
        y += 84
        f = font(25)
        for desc, ok in [(g, False) for g in hud.gone] + [(p, True) for p in hud.pool]:
            lines = wrap(d, HYPO_JA.get(desc, desc), f, W - 30 - (x0 + 36))[:2]
            if y + 34 * len(lines) > BOTTOM - 50:
                break
            d.text((x0, y), "・" if ok else "×", font=f, fill=WHITE if ok else RED)
            for ln in lines:
                d.text((x0 + 36, y), ln, font=f, fill=WHITE if ok else RED)
                if not ok:
                    d.line([x0 + 36, y + 17, x0 + 36 + d.textlength(ln, font=f), y + 17], fill=RED, width=3)
                y += 34
            y += 10
    if hud.banner and hud.gone:                    # the moment a hypothesis is thrown away
        f1, f2 = font(46), font(38)
        texts = [HYPO_JA.get(g, g) for g in hud.gone]
        w = max([d.textlength("× 観察と合わない仮説を捨てた", font=f1)] + [d.textlength(f"「{t}」", font=f2) for t in texts])
        bx, by = (PANEL - w) / 2 - 30, TOP + 110
        d.rectangle([bx, by, bx + w + 60, by + 90 + 58 * len(texts)], fill=BLACK, outline=RED, width=6)
        d.text((bx + 30, by + 18), "× 観察と合わない仮説を捨てた", font=f1, fill=RED)
        for k, t in enumerate(texts):
            d.text((bx + 30, by + 86 + 58 * k), f"「{t}」", font=f2, fill=WHITE)
    if hud.speed:
        d.rectangle([20, TOP + 20, 180, TOP + 76], fill=BLACK)
        d.text((36, TOP + 26), hud.speed, font=font(34), fill=YELLOW)
    # bottom bar: hypothesis, action and its effect, agent
    d.line([0, BOTTOM, W, BOTTOM], fill=DARK, width=2)
    d.text((30, BOTTOM + 12), "AI のことば (いまの考え)", font=font(22), fill=YELLOW)
    said = ja_said(hud.hypothesis)
    for i, ln in enumerate(wrap(d, said, font(36), W - 60)[:2]):
        d.text((30, BOTTOM + 42 + 46 * i), ln, font=font(36), fill=WHITE)
    if hud.action:
        y = BOTTOM + 136
        d.text((30, y), "行動:", font=font(28), fill=GREY)
        purpose = ja_note(hud.note)[1]
        act = hud.action + (f" ({purpose})" if purpose else "")
        d.text((112, y), act, font=font(28), fill=WHITE)
        if hud.effect:
            x = 112 + d.textlength(act, font=font(28)) + 20
            ok = hud.effect != "変化なし"
            d.text((x, y), "→ " + hud.effect, font=font(28), fill=GREEN if ok else GREY)
    d.text((PANEL + 38, BOTTOM - 40), f"AI: {hud.agent}", font=font(20, "regular"), fill=CYAN)
    return np.asarray(img)


def project(xyz, pos, lookat, fov, res=CAM_RES):
    """World point -> pixel in the camera image (pinhole, vertical fov, z up)."""
    pos, xyz = np.asarray(pos, float), np.asarray(xyz, float)
    f = np.asarray(lookat, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = xyz - pos
    foc = res[1] / 2 / np.tan(np.radians(fov) / 2)
    return res[0] / 2 + foc * (d @ r) / (d @ f), res[1] / 2 - foc * (d @ u) / (d @ f)


# ---- cards ---------------------------------------------------------------------------------------
def card(lines, y=170, strip=True, bg=None):
    """Centred card. lines: [(text, size, colour)] or [(text, size, colour, kind)]; '' adds a gap.
    `bg`: optional camera image shown dimmed behind the text."""
    img = Image.new("RGB", (W, H), BLACK)
    if bg is not None:
        img = Image.blend(Image.fromarray(np.asarray(bg)).convert("RGB").resize((W, H)), img, 0.62)
    d = ImageDraw.Draw(img)
    for item in lines:
        text, size, colour = item[:3]
        f = font(size, item[3] if len(item) > 3 else "bold")
        for ln in wrap(d, text, f, W - 240) if text else [""]:
            w = d.textlength(ln, font=f)
            d.text(((W - w) / 2, y), ln, font=f, fill=colour)
            y += int(size * 1.55)
    if strip:
        palette_strip(d, (W - 10 * 36) // 2, H - 110)
    return np.asarray(img)


def title_card(bg=None):
    return card([("PHYSICAL ARC-AGI-3", 96, WHITE, "mono"),
                 ("非公式・ARC-AGI-3 に着想を得た個人作品 (ARC Prize Foundation とは無関係)", 30, GREY, "regular"),
                 ("", 30, WHITE),
                 ("ルールを教えられていないロボットが、", 54, WHITE),
                 ("触って、試して、ルールを見つける", 54, YELLOW)], y=250, bg=bg)


def rules_card():
    return card([("ルールもゴールも教えない", 64, YELLOW),
                 ("", 20, WHITE),
                 ("ロボットにできるのは、キューブを「つかむ」と、マスに「置く」の 2 つだけ", 40, WHITE),
                 ("盤の横のランプが緑になればクリア。何をすれば点くかは秘密", 40, WHITE),
                 ("4 レベルあり、それぞれ隠れたルールが違う", 40, WHITE),
                 ("", 30, WHITE),
                 ("元ネタは ARC-AGI-3: 説明なしのゲームで、その場でルールを学ぶ力を測る AI のテスト", 32, CYAN)], y=250)


def method_card():
    img = Image.fromarray(card([("工夫: 仮説をプログラムにして確かめる", 56, YELLOW)], y=150))
    d = ImageDraw.Draw(img)
    boxes = [("1", "AI が仮説を", "Python の関数で書く"),
             ("2", "これまでの観察と", "照らし、矛盾したら捨てる"),
             ("3", "残った仮説の予想が", "割れる一手を選んで試す")]
    bw, gap, y = 480, 120, 380
    x = (W - 3 * bw - 2 * gap) // 2
    for i, (n, a, b) in enumerate(boxes):
        bx = x + i * (bw + gap)
        d.rounded_rectangle([bx, y, bx + bw, y + 240], radius=18, outline=WHITE, width=4)
        d.text((bx + 24, y + 18), n, font=font(44, "mono"), fill=YELLOW)
        for k, t in enumerate((a, b)):
            w = d.textlength(t, font=font(36))
            d.text((bx + (bw - w) / 2, y + 96 + 56 * k), t, font=font(36), fill=WHITE)
        if i < 2:
            ax = bx + bw + 20
            d.polygon([(ax, y + 100), (ax + 60, y + 120), (ax, y + 140)], fill=YELLOW)
    for k, t in enumerate(["ロボットの一手は取り消せないので、コードの上で仮説を見分けてから動く。",
                           "画面の右に AI が書いた仮説が並び、観察と合わないものに × が付く。"]):
        w = d.textlength(t, font=font(34, "regular"))
        d.text(((W - w) / 2, 720 + 58 * k), t, font=font(34, "regular"), fill=WHITE)
    return np.asarray(img)


def guide_card(frame):
    """A HUD frame, dimmed, with the four regions a viewer should read highlighted and labelled."""
    src = Image.fromarray(np.asarray(frame)).convert("RGB")
    img = Image.blend(src, Image.new("RGB", (W, H), BLACK), 0.6)
    d = ImageDraw.Draw(img)
    regions = [((20, TOP + 20, PANEL - 20, BOTTOM - 20), "ロボットと盤面 (物理シミュレーション)", (420, TOP + 50)),
               ((PANEL + 20, TOP + 8, W - 20, BOTTOM - 10), "AI に渡す盤面と、AI が書いた仮説", (720, TOP + 250)),
               ((10, BOTTOM + 4, W - 10, H - 4), "AI のことばと、いまの行動・結果", (1100, BOTTOM + 30)),
               ((940, 4, W - 20, TOP - 4), "行動回数 (少ないほど良い) とランプ (緑ならクリア)", (560, TOP + 140))]
    for box, _, _ in regions:
        img.paste(src.crop(box), box[:2])
    for box, label, (lx, ly) in regions:
        d.rectangle(box, outline=YELLOW, width=5)
        f = font(34)
        tw = d.textlength(label, font=f)
        d.rectangle([lx - 12, ly - 8, lx + tw + 12, ly + 48], fill=YELLOW)
        d.text((lx, ly), label, font=f, fill=BLACK)
    t = "画面の見方"
    d.rectangle([20, TOP + 20, 20 + d.textlength(t, font=font(56)) + 40, TOP + 110], fill=BLACK)
    d.text((40, TOP + 30), t, font=font(56), fill=WHITE)
    return np.asarray(img)


def level_intro(n, n_levels):
    sub = ("ルールは秘密。ランプを緑にできたらクリア" if n == 1 else "今度の手がかりは、見ただけでは分からない" if n == n_levels
           else "新しいルール。前のレベルのやり方はそのままでは通用しない")
    return card([(f"LEVEL {n} / {n_levels}", 110, WHITE, "mono"), ("", 20, WHITE), (sub, 44, YELLOW)], y=330)


def level_card(n, actions, optimal, rule, last_hypothesis, remark="", solved=True):
    return card([(f"LEVEL {n} クリア!" if solved else f"LEVEL {n} 未クリア", 84, GREEN if solved else RED),
                 (f"行動回数 {actions}  (最短 {optimal})" if solved else f"上限の {actions} 手で打ち切り (最短 {optimal})", 46, YELLOW),
                 ("", 24, WHITE),
                 ("隠れていたルール (AI には最後まで知らされない):", 32, GREY, "regular"),
                 (rule, 42, WHITE),
                 ("", 20, WHITE),
                 ("AI が最後に立てていた仮説:", 32, GREY, "regular"),
                 (f"「{ja_said(last_hypothesis)}」", 34, CYAN, "regular"),
                 ("", 20, WHITE),
                 (remark, 34, YELLOW)], y=150)


def table_card(rows, takeaway):
    """End card. rows: [(level, actions or None, random-baseline text, optimal)]."""
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    d.text((100, 70), "結果: クリアまでの行動回数 (少ないほど良い)", font=font(52), fill=WHITE)
    cols = [100, 420, 760, 1140]
    for x, t in zip(cols, ["レベル", "今回の AI", "最短", "でたらめに動かすと"]):
        d.text((x, 200), t, font=font(34), fill=YELLOW)
    d.line([100, 256, W - 100, 256], fill=GREY, width=2)
    for i, (lv, mine, rnd, opt) in enumerate(rows):
        y = 280 + i * 100
        d.text((cols[0] + 30, y), str(lv), font=font(60, "mono"), fill=WHITE)
        d.text((cols[1], y), "未クリア" if mine is None else f"{mine} 手", font=font(56),
               fill=RED if mine is None else GREEN)
        d.text((cols[2], y), f"{opt} 手", font=font(56), fill=GREY)
        d.text((cols[3], y + 12), rnd, font=font(34, "regular"), fill=WHITE)
    for k, t in enumerate(takeaway):
        d.text((100, 280 + 100 * len(rows) + 40 + 60 * k), t, font=font(34, "regular"), fill=WHITE if k == 0 else GREY)
    palette_strip(d, 100, H - 100)
    return np.asarray(img)


def physics_card():
    """Before the weight level: what the arm can feel and how imagination turns torques into a weight."""
    img = Image.fromarray(card([("工夫 2: 触って、想像の世界と比べる", 56, YELLOW)], y=130))
    d = ImageDraw.Draw(img)
    lines = [("見た目が同じでも、重さは持ち上げないと分からない。", WHITE),
             ("持ち上げると関節にかかる力 (トルク) が変わる。ただし、その値は腕の姿勢やつかみ方でも変わる。", WHITE),
             ("そこで、重さだけを変えた 6 つの「想像の世界」(シミュレーション) で同じ動きを試し、", WHITE),
             ("実際の力に一番近い世界の重さを、そのキューブの重さとする。", GREEN),
             ("", WHITE),
             ("AI の仮説も重さを使って書ける。答えが重さしだいなら、まず持ち上げて測る。", CYAN)]
    y = 300
    for t, c in lines:
        if t:
            w = d.textlength(t, font=font(36, "regular"))
            d.text(((W - w) / 2, y), t, font=font(36, "regular"), fill=c)
        y += 70
    return np.asarray(img)


def imagine_frame(base, view, masses, xs, taus=None, real=None, best=None, mass=None, cube=""):
    """The camera area replaced by the imagined worlds. `base` is the last HUD frame, `view` one imagination
    render, `xs` the x pixel of each world. With `taus`, shows how far each world's torques are from the real ones."""
    img = Image.fromarray(np.asarray(base)).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, TOP, PANEL, BOTTOM], fill=BLACK)
    d.text((30, TOP + 14), f"想像の世界: {COLOUR_JA.get(cube, cube)}のキューブの重さを変えた 6 つのシミュレーション", font=font(36), fill=YELLOW)
    if taus is not None:
        d.text((30, TOP + 62), f"→ 実際の力に一番近い世界: {COLOUR_JA.get(cube, cube)}のキューブは約 {mass:.2f} kg", font=font(32), fill=GREEN)
    else:
        d.text((30, TOP + 66), "実際と同じ姿勢から同じ動きでつかみ上げ、関節にかかる力を比べる", font=font(26, "regular"), fill=WHITE)
    vh = np.asarray(view).shape[0]
    img.paste(Image.fromarray(np.asarray(view)).convert("RGB"), (0, TOP + 110))
    y = TOP + 110 + vh + 12
    errs = None if taus is None else [float(np.linalg.norm(t - real)) for t in taus]
    for i, (m, x) in enumerate(zip(masses, xs)):
        t = f"{m:g} kg"
        f = font(34, "mono")
        tw = d.textlength(t, font=f)
        colour = GREEN if best == i else WHITE
        d.text((x - tw / 2, y), t, font=f, fill=colour)
        if errs is not None:
            h = int(min(errs[i] / max(errs), 1) * 70)
            d.rectangle([x - 18, y + 116 - h, x + 18, y + 116], fill=GREEN if best == i else GREY)
        if best == i:
            d.rectangle([x - 110, TOP + 110, x + 110, TOP + 110 + vh], outline=GREEN, width=6)
    if errs is not None:
        d.text((30, y + 64), "実際の力との", font=font(22, "regular"), fill=GREY)
        d.text((30, y + 92), "ずれ", font=font(22, "regular"), fill=GREY)
    return np.asarray(img)
