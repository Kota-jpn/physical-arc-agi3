# Physical ARC-AGI-3

> **非公式・ARC-AGI-3 に着想を得たもの / unofficial, inspired by ARC-AGI-3, not affiliated with ARC Prize Foundation.**
> 本作品は個人の課題作品であり、ARC Prize Foundation の公式製品・公式ベンチマークではありません。公式のゲームやデータは使っていません。
> 着想元: <https://arcprize.org/arc-agi/3/>

**English summary.** An unofficial, ARC-AGI-3-inspired environment built in the Genesis simulator (CPU only). A Franka Panda arm plays four turn-based levels on a 6x6 board of ARC-palette tiles with colour cubes. The agent is *not* told the rules or the goal; each turn it receives the board as an integer grid, the lamp state, the cubes and its action history, and returns a short Japanese hypothesis plus one action, `pick(object)` or `place(cell)`. The hidden rules are colour matching (L1), a switch tile that reveals hidden target colours (L2), counting (L3), and a hidden physical property (L4: two identical-looking cubes are heavy; the pressure pad needs enough total mass). The arm feels weight when lifting: its joint torques are compared with K parallel simulated copies of the same pick that differ only in the cube mass ("imagination"), and LLM hypotheses may use the felt weights; when a prediction hinges on an unweighed cube, the planner lifts it first. Three LLM conditions share one model: prompt-only (`llm-plain`), cheap tricks borrowed from ARC-AGI-3 Kaggle Milestone 1 winners (`llm-tricks`), and our `llm-hypo`, where the LLM writes hypotheses as Python predicates that code falsifies locally and then chooses the experiment that best splits the surviving hypotheses (think in simulation, then act). The LLM is Gemini through Google's OpenAI-compatible endpoint (inference only, no training). The game logic is pure Python and separate from the physics: the game state is derived from the observed cube poses by snapping them to grid cells. Without an API key a clearly labelled, hand-coded offline heuristic (not an LLM) runs instead. A symbolic evaluator compares agents with a random baseline. Course: Physical AI 2026, final project.

## コンセプト

ARC-AGI-3 は、説明もルールも目標も与えられない対話型のターン制環境で、エージェントが自分で探索して規則を推測し、少ない行動数で解けるかを測るベンチマークです (公式の技術報告: <https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf>)。
この課題では、その発想をロボットアームの卓上に持ち込みました。アームは「ルールもゴールも教えられない」まま、キューブを掴んで盤面に置き、返ってくる視覚フィードバックから隠れたルールを推測して、ランプを点灯させます。重みの更新は行わず、推論だけでテスト時に適応させます。

## レベル

盤面は 6x6。床タイルは ARC の標準 10 色 (0 黒, 1 青, 2 赤, 3 緑, 4 黄, 5 灰, 6 マゼンタ, 7 オレンジ, 8 水色, 9 栗色)、動かせる物は単色キューブ (red / blue / green / yellow) です。盤面の脇のランプは灰色で、レベルを解くと緑に点灯します。
行動は `pick(object_id)` と `place(cell=[row,col])` の 2 種類で、占有セルへの place は積み上げになります (最大 4 段)。実行した行動はすべて数えます。

| レベル | 隠れたルール (エージェントには非公開) | 見えるフィードバック | 最短 |
|---|---|---|---|
| L1 色合わせ | 同じ色のタイルにキューブを置くとそのタイルが点灯。3 枚すべてでランプ点灯 (黄キューブは囮) | 白い枠が付く | 6 |
| L2 因果スイッチ | オレンジのスイッチ上にキューブがある間だけ灰色タイルが色を現す。現れた色に合わせる | 灰色が色に変わる / 枠 | 6 |
| L3 数え上げ | マーカー (マゼンタ) の数 N と同じ高さの塔を水色タイルに作る | 積むたびにマーカーが 1 つずつ点灯 | 6 |
| L4 重さ | 見た目は同じでも青と黄のキューブだけが重い (0.3 kg、他は 0.04 kg)。栗色の 2 マスのパッドに合計 0.5 kg 以上を載せる | パッドに何か載ると枠。重さは持ち上げたときに腕で感じる | 4 |

## セットアップ (macOS / Linux, CPU で動作)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
cp .env.example .env     # Gemini を使う場合のみ。.env は git 管理外
```

依存は `requirements.txt` でバージョン固定しています (genesis-world 1.4.4, torch 2.14.1, openai 3.31.0 ほか)。

### `.env`

```
GEMINI_API_KEY=<your-key>
GEMINI_MODEL=gemini-3.1-flash-lite      # 任意。未設定時もこの値
# PAI_OFFLINE=1                         # キーがあってもオフラインのヒューリスティックを強制
```

Google の OpenAI 互換エンドポイント (`https://generativelanguage.googleapis.com/v1beta/openai/`) を `openai` パッケージで呼びます。
無料枠にはモデルごとの 1 日あたりの上限があります。使い切ると 429 が続くので、その場合は `GEMINI_MODEL` を変えてください。

## エージェントの 3 条件

同じモデル (既定 `gemini-3.1-flash-lite`、無料枠) で、LLM の使い方だけを変えた 3 条件を比べます。

| `--agent` | 内容 |
|---|---|
| `llm-plain` | プロンプトのみ。観測・実験メモ・前レベルの教訓を渡し、構造化した仮説 (競合する 2-3 個) と 1 手を JSON で返させる |
| `llm-tricks` | `llm-plain` に、ARC-AGI-3 Kaggle Milestone 1 上位チームの安い工夫を足したもの: 直前の差分要約、効果のなかった行動の記録、既訪問状態の警告、1-4 手のまとめ出し、引き継ぐノート |
| `llm-hypo` | 本課題の提案。LLM が仮説を Python 関数 `lit(state)` (ランプが点く条件) として書き、コードが観測済みの全状態に対して手元で反証する (API 費用なし)。次の行動は、生き残った仮説の予測を最も割る状態へ至る動きから選び、残りの仮説が一致したらその計画を実行する。シンボリックなモデルは物理の遷移 (キューブの行き先) のシミュレータとしてだけ使い、ランプの正解 oracle としては使わない (`src/pai/hypo.py`) |

ロボットの行動は高コストで取り消せないので、「シミュレーションで仮説を見分けてから動く」ことを狙った設計です。

### 重さを感じる: 想像の世界との照合 (`src/pai/imagine.py`)

キューブを持ち上げると関節トルクが変わりますが、その値は腕の姿勢やつかみ方にも依存します。そこで手書きの換算式は使わず、実際のピック直前の腕の状態とキューブの位置から、同じピック動作を K=6 個の並列シミュレーション (1 つの Genesis シーンの並列環境、質量だけ 0.04〜0.8 kg で違う) で再生し、関節トルクが実測に最も近い世界 (隣り合う世界の間は線形補間) の質量を「感じた重さ」とします。真の質量は読みません。感じた重さは履歴と `state.mass(cube)` で LLM の仮説に渡り、答えがまだ持ち上げていないキューブの重さしだいなら、プランナーはそのキューブを持ち上げる手を実験として選びます (`hypo.search` の `weigh`)。物理面では、重いキューブが軽いキューブの上で滑る・指の中でねじれるのを防ぐため、Genesis の no-slip 反復とねじり摩擦を有効にしています。

## 実行

### デモ (物理シミュレーション + 動画)

```bash
./run.sh --offline --video outputs/demo_offline.mp4   # キー無し: オフラインのヒューリスティック (LLM ではない)
./run.sh --agent llm-hypo --cap 30 --speed 4 --video outputs/demo.mp4   # キーあり: Gemini (--agent の既定は llm-hypo)
./run.sh --offline --no-render                        # 動画なしでロジックだけ確認
```

`run.sh` (`pai.run`) のオプション: `--video PATH`, `--agent {llm-plain,llm-tricks,llm-hypo}` (既定 llm-hypo), `--offline`, `--no-render`, `--speed N` または `--speed N,N,N` (動画の再生倍率。レベルごとに指定可。既定 4。画面に `N倍速` と表示), `--cap N` (レベルあたりの行動数上限。既定 40), `--replay agent_log.jsonl` (過去の実行で実際に行われた判断を再生して動画を作り直す。API は呼ばない), `--remark N=TEXT` (レベル N のクリア画面に一文を足す), `--levels N ...` (一部のレベルだけ実行)。

提出動画は次のコマンドで作り直したものです (1920x1080、AI は呼ばない):

```bash
PYTHONPATH=src python -m pai.run --replay outputs/agent_log_demo_spliced.jsonl --cap 30 --speed 4,6,8,6 \
  --remark '3=このレベルだけ、点いた数と一致する量を AI に見せる版で撮り直した。AI の予想は 4 段だった。' \
  --remark '4=AI は持ち上げて測った重さ (青 0.30 / 黄 0.20 / 緑 0.10 / 赤 0.04 kg) を使って仮説を立てていた。' \
  --video outputs/demo.mp4
```
キーが無い、または `--offline` / `PAI_OFFLINE` が指定されたときは手書きのオフライン・ヒューリスティックが動き、動画の HUD に `agent: offline heuristic` と表示されます。

### ナレーション (任意)

提出動画の日本語ナレーションは VOICEVOX (No.7) で読み上げています。ローカルの VOICEVOX エンジン (port 50021) が必要です。

```bash
python scripts/narrate.py synth video/narration.json          # 読み上げ -> video/tts/, カードの長さ video/cards.json
PYTHONPATH=src python -m pai.run --replay outputs/agent_log_demo_spliced.jsonl --cards video/cards.json ... --video outputs/demo_silent.mp4
python scripts/narrate.py mix video/narration.json outputs/demo_silent.mp4 outputs/demo.mp4
```

### 重さの推定の頑健さ

```bash
PYTHONPATH=src python scripts/weigh_robustness.py             # -> outputs/weigh_robustness.json
```

「実際の世界」の側だけにトルクの雑音、つかむ力の違い、手の想定外の重さを入れ、想像の世界との照合で重さがどれだけずれるかを測ります。

### 記号評価 (物理なし、高速)

```bash
./eval.sh --agent random                                  # 既定 200 エピソード/レベル, 上限 200 行動
./eval.sh --agent heuristic
./eval.sh --agent llm-hypo --episodes 3 --cap 40          # 要 GEMINI_API_KEY。LLM の既定は 3 エピソード
./eval.sh --agent llm-tricks --episodes 3 --cap 40
./eval.sh --agent llm-plain --episodes 2 --cap 40
```

`--agent {random,heuristic,llm-plain,llm-tricks,llm-hypo}` (必須), `--episodes N`, `--cap N` (既定 200), `--levels N ...` (一部のレベルだけを試す。保存されない)。
結果は `outputs/baseline.json` にエージェント別に保存されます。無料枠の 1 日上限に達すると、終わったエピソードまでを残して終了コード 3 で止まります。

### 最終実行 (まとめて)

```bash
tmux new-session -d -s final 'scripts/final_run.sh > outputs/final_run.log 2>&1'
```

`scripts/final_run.sh` は、llm-hypo の物理デモ (`outputs/demo.mp4`、長さに応じて再生倍率を調整して再描画)、説明資料用のフレーム (`outputs/frames/`)、3 条件の記号評価 (cap 40) を順に実行します。前回の成果物は `outputs/old/` に退避します。
無料枠 (gemini-3.1-flash-lite は 1 日 500 リクエスト) の範囲で収まる想定です。

### 出力

`outputs/` (git 管理外。`demo.mp4` のみ追跡): `demo.mp4`, `baseline.json`, `demo_run.json` (物理実行の結果), `demo_timeline.json`, `frames/`, `agent_log.jsonl` (LLM への全リクエスト/レスポンスと実行した判断。429 の再試行も記録)。

### 説明資料 (PDF)

```bash
report/build.sh    # 動画から画像を切り出し、report.md を pandoc で HTML にして Chrome headless で PDF を生成
```

`pandoc`、`ffmpeg`、Google Chrome が必要です。画像は `outputs/demo.mp4` から `outputs/demo_timeline.json` の時刻で切り出します。

## ベースライン

記号評価 (2026-10-24 の最終実行, `gemini-3.1-flash-lite`, LLM 条件は上限 40 行動。値は到達率 / 到達時の平均行動数。詳細は `outputs/baseline.json`):

| エージェント | L1 | L2 | L3 |
|---|---|---|---|
| ランダム (200 エピソード, 上限 200) | 1% / 108 | 0.5% / 46 | 0.5% / 110 |
| オフライン・ヒューリスティック (200 エピソード) | 100% / 6 | 100% / 8 | 100% / 6 |
| llm-plain (2 エピソード) | 50% / 6 | 100% / 6 | 50% / 20 |
| llm-tricks (3 エピソード) | 67% / 6 | 33% / 6 | 0% / — |
| llm-hypo (3 エピソード) | 100% / 6 | 67% / 29 | 0% / — |
| 最短 | 6 | 6 | 6 |

オフライン・ヒューリスティックは、規則の族と探索順序を人間が手で書き込んだ参照値であり、**LLM ではありません**。LLM 条件はエピソード数が少なく、条件間の優劣は結論できません。L1〜L3 の表は L4 を足す前のコードで測ったものです。
`outputs/baseline.json` の `llm-hypo` の項目は、その後 L1〜L4 を続けて回した修正後の評価 (上限 30 行動、2026-10-28 の 4 エピソードと 10-29 の 5 エピソードを合算) で上書きされています。

L4 (天秤) は 2026-10-28 に、L1〜L4 を続けて解く評価で測りました (上限 30 行動。`--no-weight` は感じた重さを観測から消すアブレーション)。「修正前/後」は、感じた重さどうしの `==` を測定の分解能 (0.015 kg) 以内の一致として扱う修正 (`hypo.Kg`) の前後です:

| エージェント | L4 |
|---|---|
| ランダム (200 エピソード, 上限 200) | 0% |
| オフライン・ヒューリスティック | 0% |
| llm-hypo, 重さなし (5 エピソード) | 0/5 |
| llm-hypo, 重さあり, 修正前 (5 エピソード) | 1/5 (28 行動) |
| llm-hypo, 重さあり, 修正後 (9 エピソード: 10-28 に 4 回 (5 回目はクォータで打ち切り) + 10-29 に 5 回) | 5/9 (13 / 14 / 27 / 29 / 30 行動) |

```bash
./eval.sh --agent llm-hypo --levels 4 --episodes 5 --cap 30
./eval.sh --agent llm-hypo --levels 4 --episodes 5 --cap 30 --no-weight
```

L3 (数え上げ) で色合わせから抜け出せない問題に対して、2026-11-01 に、観測したすべての状態で「枠の付いたセルの数」と常に一致した単純な量 (あるセルの高さ、同色タイル上のキューブ数など) と、色ごとのタイル枚数をプロンプトに添える手がかりを足しました (`LLMAgent._regularities`、レベル固有の知識は使わない)。同じ評価 (L1〜L4 通し、上限 30、5 エピソード、`baseline.json` の `llm-hypo-regularities`) の到達数は L1 4/5, L2 2/5, L3 3/5, L4 4/5 です (手がかりなしの 9 エピソードは 9/9, 3/9, 1/9, 5/9)。

物理デモ (`outputs/demo.mp4`) は 2 回の実行をつないでいます: L1/L2/L4 は 2026-10-28 の実行 (6/22/26 行動。このとき L3 は 30 行動で解けず)、L3 は手がかりを足した後の 2026-11-01 の実行 (18 行動。この実行は L1〜L4 を順に 6/22/18/21 行動で解いた。`outputs/agent_log_live7.jsonl`)。つないだ判断ログは `outputs/agent_log_demo_spliced.jsonl` で、動画の L3 クリア画面にも撮り直しを表示しています。

## 構成

```
src/pai/grid.py      盤面座標と ARC パレット (セル <-> ワールド座標, 姿勢 -> セル・段数)
src/pai/game.py      ゲームロジック (純 Python): 4 レベルの隠れルール, フィードバック, ランプ, 行動の検証, 履歴
src/pai/agents.py    random / オフラインヒューリスティック / Gemini の 3 条件 (plain, tricks, hypo) / ログ再生
src/pai/hypo.py      llm-hypo: 仮説プログラムの実行・反証・実験選択 (遷移シミュレータのみ。ランプの正解は呼ばない)
src/pai/scene.py     Genesis シーン (タイルとランプは見た目のみ) と Franka 制御 (IK + 摩擦把持, 持ち上げ時の関節トルク)
src/pai/imagine.py   想像の世界: 質量だけ違う並列シミュレーションで同じピックを再生し、トルク照合で重さを推定
src/pai/run.py       物理実行 + 動画収録 (姿勢を観測 -> ルール判定 -> 表示)
src/pai/evaluate.py  物理なしの評価 (random / heuristic / llm-plain / llm-tricks / llm-hypo)
src/pai/overlay.py   ARC 風 HUD・タイトル・LEVEL COMPLETE・結果カード
scripts/final_run.sh 最終実行 (デモ動画, フレーム, 3 条件の評価)
run.sh / eval.sh     実行ラッパ (.venv を有効化し PYTHONPATH=src で呼ぶ)
report/              説明資料: report.md -> report.html -> PDF, style.css, build.sh, img/
outputs/             動画と結果 (上記)
```

## 設計メモ

物理とゲームロジックは分離しています。ロボットは行動を実行するだけで、実行後にキューブの姿勢を読み、セルにスナップした状態を `game.py` に渡します。ランプやタイルの表示は `game.py` の返り値をそのまま描いたものです。
LLM の出力は JSON で受け、行動を盤面の状態に対して検証し、不正なら理由を返して最大 3 回まで再生成させます (llm-hypo では仮説コードも制限付き実行で検証します)。無料枠の 429 / 5xx / タイムアウトは指数バックオフで再試行します。
API キーはコードに含めず、環境変数か `.env` から読みます。LLM の応答は実行ごとに変わり得ますが、物理シミュレーションは決定的です。
