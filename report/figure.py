"""Draws img/f_method.png: the loop of the agent (for report.md). Run from report/: PYTHONPATH=../src python figure.py"""
from PIL import Image, ImageDraw

from pai.overlay import font

W, H = 1800, 860
INK, MUTE, BLUE, GREEN, ORANGE = (30, 30, 30), (110, 110, 110), (0, 116, 217), (30, 150, 60), (230, 120, 20)
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)


def box(x, y, w, h, title, lines, colour):
    d.rounded_rectangle([x, y, x + w, y + h], radius=18, outline=colour, width=5)
    d.text((x + 24, y + 18), title, font=font(34), fill=colour)
    for k, t in enumerate(lines):
        d.text((x + 24, y + 72 + 40 * k), t, font=font(27, "regular"), fill=INK)


def arrow(x0, y0, x1, y1, label="", colour=INK):
    d.line([x0, y0, x1, y1], fill=colour, width=5)
    if x1 != x0:
        s = 1 if x1 > x0 else -1
        d.polygon([(x1, y1), (x1 - 22 * s, y1 - 12), (x1 - 22 * s, y1 + 12)], fill=colour)
    else:
        s = 1 if y1 > y0 else -1
        d.polygon([(x1, y1), (x1 - 12, y1 - 22 * s), (x1 + 12, y1 - 22 * s)], fill=colour)
    if label:
        d.text(((x0 + x1) / 2 - d.textlength(label, font=font(24, "regular")) / 2, max(y0, y1) + 14), label,
               font=font(24, "regular"), fill=MUTE)


box(40, 60, 500, 250, "1. 観察", ["盤面 (色の数字の表)", "枠・ランプ・行動の記録", "持ち上げて感じた重さ"], INK)
box(650, 60, 500, 250, "2. AI が仮説を書く", ["Gemini が「ランプが点く条件」を", "Python の関数で 2〜4 個", "例: 2 つの皿の重さが等しい"], BLUE)
box(1260, 60, 500, 250, "3. 手元で反証", ["これまでの全盤面に当てはめ、", "結果と食い違う仮説を捨てる", "(AI を呼ばない)"], BLUE)
box(1260, 470, 500, 300, "4. 次の一手を選ぶ", ["6 手先まで探して", "・予想が割れる盤面 → 試す", "・全員が点灯と予想 → 解く", "・重さしだい → 持ち上げて測る"], GREEN)
box(650, 470, 500, 300, "5. ロボットが実行", ["Genesis 上の Franka が", "つかむ・置く", "結果はキューブの実際の位置から", "盤面に直して判定"], INK)
box(40, 470, 500, 300, "想像の世界 (持ち上げたとき)", ["持ち上げる直前の状態を写した", "6 つの並列シミュレーション", "(重さだけ 0.04〜0.8 kg)", "関節の力が一番近い世界の重さ"], ORANGE)
arrow(540, 185, 650, 185)
arrow(1150, 185, 1260, 185)
arrow(1510, 310, 1510, 470)
arrow(1260, 620, 1150, 620)
arrow(650, 620, 540, 620)
d.text((306, 372), "感じた重さ", font=font(24, "regular"), fill=ORANGE)
arrow(290, 470, 290, 310, "", ORANGE)
d.line([900, 470, 900, 400, 440, 400], fill=INK, width=5)          # result of the action -> next observation
arrow(440, 400, 440, 310, "")
d.text((560, 360), "結果 (枠・ランプ・キューブの位置)", font=font(24, "regular"), fill=MUTE)
img.save("img/f_method.png")
print("img/f_method.png")
