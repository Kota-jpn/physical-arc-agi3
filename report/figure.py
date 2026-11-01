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

# ---- img/f_arch.png: components and what flows between them ---------------------------------------
W, H = 1800, 960
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)
Y = 70                                              # room above the groups for the action arrow


def group(x, y, w, h, label, colour):
    d.rounded_rectangle([x, y, x + w, y + h], radius=24, outline=colour, width=3, fill=(248, 248, 248))
    d.text((x + 20, y + 12), label, font=font(28), fill=colour)


group(20, Y, 560, 600, "物理シミュレーション (Genesis, CPU)", INK)
group(640, Y, 480, 600, "ゲーム (純 Python)", INK)
group(1200, Y, 580, 600, "エージェント (llm-hypo)", BLUE)
box(50, Y + 60, 500, 230, "ロボットと盤 scene.py", ["Franka Panda、6×6 の盤", "キューブ 4 個 (重さは非公開)", "IK で動かし、摩擦でつかむ"], INK)
box(50, Y + 350, 500, 230, "想像の世界 imagine.py", ["重さだけ違う 6 つの並列環境で", "同じ持ち上げ動作を再生し、", "関節トルクを実測と比べる"], ORANGE)
box(665, Y + 60, 430, 230, "観測の変換 grid.py", ["キューブの実際の位置を", "一番近いマスに当てはめる"], INK)
box(665, Y + 350, 430, 230, "ルール判定 game.py", ["隠れたルール (4 レベル)", "枠・タイルの色・ランプ", "物理なしの評価にも使う"], INK)
box(1225, Y + 60, 530, 230, "Gemini (推論のみ)", ["仮説を Python の関数で書く", "いまの考えを言葉で書く"], BLUE)
box(1225, Y + 350, 530, 230, "反証と実験選択 hypo.py", ["観察と食い違う仮説を捨てる", "6 手先まで探し、予想が割れる手・", "解く手・重さを測る手を選ぶ"], BLUE)
box(20, 760, 820, 180, "記録 agent_log.jsonl", ["AI とのやり取りと実行した判断をすべて記録", "記録から AI を呼ばずに同じ判断を再生できる"], MUTE)
box(900, 760, 880, 180, "動画 run.py --replay + overlay.py", ["記録を再生して物理を描き直し、説明を重ねる", "ナレーションは VOICEVOX で読み上げ"], MUTE)
arrow(550, Y + 175, 665, Y + 175)
d.text((572, Y + 130), "位置", font=font(24, "regular"), fill=MUTE)
arrow(880, Y + 290, 880, Y + 350)
d.text((896, Y + 302), "盤面", font=font(24, "regular"), fill=MUTE)
arrow(1095, Y + 465, 1225, Y + 465)
d.text((1128, Y + 420), "観測", font=font(24, "regular"), fill=MUTE)
arrow(1420, Y + 350, 1420, Y + 290, "", BLUE)
arrow(1560, Y + 290, 1560, Y + 350, "", BLUE)
d.text((1340, Y + 302), "依頼", font=font(22, "regular"), fill=BLUE)
d.text((1576, Y + 302), "仮説のコード", font=font(22, "regular"), fill=BLUE)
arrow(300, Y + 290, 300, Y + 350, "", ORANGE)
d.text((316, Y + 302), "持ち上げ時のトルク", font=font(22, "regular"), fill=ORANGE)
d.line([550, Y + 540, 600, Y + 540, 600, Y + 630, 1490, Y + 630, 1490, Y + 600], fill=ORANGE, width=5)   # felt weight
d.polygon([(1490, Y + 580), (1478, Y + 602), (1502, Y + 602)], fill=ORANGE)
d.text((1000, Y + 638), "感じた重さ", font=font(22, "regular"), fill=ORANGE)
d.line([1755, Y + 465, 1790, Y + 465, 1790, 30, 610, 30, 610, Y + 110, 572, Y + 110], fill=GREEN, width=5)   # action -> robot
d.polygon([(550, Y + 110), (572, Y + 98), (572, Y + 122)], fill=GREEN)
d.text((860, 0), "行動: つかむ / 置く", font=font(24), fill=GREEN)
d.line([1300, Y + 600, 1300, 742, 430, 742, 430, 745], fill=MUTE, width=4)     # agent -> log
d.polygon([(430, 762), (418, 740), (442, 740)], fill=MUTE)
d.text((560, 708), "やり取りと判断", font=font(22, "regular"), fill=MUTE)
arrow(840, 850, 900, 850, "", MUTE)
img.save("img/f_arch.png")
print("img/f_arch.png")
