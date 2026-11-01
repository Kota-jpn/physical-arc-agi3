"""Japanese narration for the demo video (VOICEVOX, optional; needs a local VOICEVOX engine on port 50021).

  python scripts/narrate.py synth video/narration.json   # speak every line -> video/tts/*.wav, video/cards.json
  python -m pai.run --replay ... --cards video/cards.json --video outputs/demo_silent.mp4
  python scripts/narrate.py mix video/narration.json outputs/demo_silent.mp4 outputs/demo.mp4

narration.json: [{"at": anchor, "lines": ["...", {"sub": "...", "vv": "reading"}], "card": true}]
`at` is a card name from outputs/demo_timeline.json ("title", "arc", "l4_clear", ...) or an event key there
("l2_start", "imagination"), optionally "+seconds". With "card": true the card is lengthened to fit the lines.
"""
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ENGINE = "http://127.0.0.1:50021"
SPEAKER, SPEED, SR = 30, 1.15, 24000           # 30 = No.7 (アナウンス)
GAP, PAD = 0.25, 0.8                            # pause between lines, silence after a card's last line [s]
ROOT = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"


def api(path, data=None, **params):
    url = ENGINE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {},
                                 method="POST" if path != "/speakers" else "GET")
    return urllib.request.urlopen(req).read()


def spoken(line):
    return line["vv"] if isinstance(line, dict) and "vv" in line else re.sub(r"[『』「」]", "", line if isinstance(line, str) else line["sub"])


def seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def synth(script):
    blocks, tts = json.loads(script.read_text()), script.parent / "tts"
    tts.mkdir(exist_ok=True)
    cards = {}
    for b, block in enumerate(blocks):
        total = 0.4
        for k, line in enumerate(block["lines"]):
            q = json.loads(api("/audio_query", text=spoken(line), speaker=SPEAKER))
            q.update(speedScale=SPEED, outputSamplingRate=SR, outputStereo=False, prePhonemeLength=0.05, postPhonemeLength=0.05)
            out = tts / f"{b:02d}_{k:02d}.wav"
            out.write_bytes(api("/synthesis", data=json.dumps(q).encode(), speaker=SPEAKER))
            total += seconds(out) + GAP
            print(f"{block['at']:>14} {seconds(out):5.2f}s  {spoken(line)}")
        if block.get("card"):
            cards[block["at"]] = round(total + PAD, 2)
    (script.parent / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False))
    print("card lengths:", cards)


def anchor(at, tl):
    name, _, plus = at.partition("+")
    marks = {m[0]: m[1] for m in tl.get("cards", [])}
    t = marks[name] if name in marks else tl[name]
    return t + float(plus or 0)


def mix(script, silent, out):
    blocks, tts = json.loads(script.read_text()), script.parent / "tts"
    tl = json.loads((ROOT / "outputs" / "demo_timeline.json").read_text())
    clips, end = [], 0.0
    for b, block in enumerate(blocks):
        t = anchor(block["at"], tl) + 0.4
        if t < end:
            print(f"warning: '{block['at']}' starts {end - t:.1f}s before the previous line ends; delayed")
            t = end
        for k in range(len(block["lines"])):
            clip = tts / f"{b:02d}_{k:02d}.wav"
            clips.append((t, clip))
            t += seconds(clip) + GAP
        end = t
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(silent)]))
    if end > duration:
        print(f"warning: narration runs {end - duration:.1f}s past the end of the video")
    pcm = bytearray(int(duration * SR) * 2)
    for t, clip in clips:
        with wave.open(str(clip)) as w:
            data = w.readframes(w.getnframes())
        i = int(t * SR) * 2
        pcm[i:i + len(data)] = data[:len(pcm) - i]
    wav = script.parent / "narration.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(bytes(pcm))
    credit = script.parent / "credit.png"            # the voice credit required by the VOICEVOX terms, bottom right
    img = Image.new("RGBA", (300, 34), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 20)
    d.text((300 - 8 - d.textlength("VOICEVOX:No.7", font=f), 4), "VOICEVOX:No.7", font=f, fill=(170, 170, 170, 255))
    img.save(credit)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(wav), "-i", str(credit),
                    "-filter_complex", "[0:v][2:v]overlay=W-w-16:H-h-6[v]", "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    print(f"wrote {out} ({duration:.1f}s, narration ends at {end:.1f}s)")


if __name__ == "__main__":
    cmd, script = sys.argv[1], Path(sys.argv[2])
    synth(script) if cmd == "synth" else mix(script, Path(sys.argv[3]), Path(sys.argv[4]))
