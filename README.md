# Physical ARC-AGI-3

> **非公式・ARC-AGI-3 に着想を得たもの / unofficial, inspired by ARC-AGI-3, not affiliated with ARC Prize Foundation.**
> 本作品は個人の課題作品であり、ARC Prize Foundation の公式製品・公式ベンチマークではありません。公式のゲームやデータは使っていません。
> 着想元: <https://arcprize.org/arc-agi/3/>

**English summary.** An unofficial, ARC-AGI-3-inspired environment built in the Genesis simulator (CPU only). A Franka Panda arm plays three turn-based levels on a 6x6 board of ARC-palette tiles with colour cubes. The agent is *not* told the rules or the goal; each turn it receives the board as an integer grid, the lamp state, the cubes and its action history, and returns a short Japanese hypothesis plus one action, `pick(object)` or `place(cell)`. The hidden rules are colour matching (L1), a switch tile that reveals hidden target colours (L2), and counting (L3). Three LLM conditions share one model: prompt-only (`llm-plain`), cheap tricks borrowed from ARC-AGI-3 Kaggle Milestone 1 winners (`llm-tricks`), and our `llm-hypo`, where the LLM writes hypotheses as Python predicates that code falsifies locally and then chooses the experiment that best splits the surviving hypotheses (think in simulation, then act). The LLM is Gemini through Google's OpenAI-compatible endpoint (inference only, no training). The game logic is pure Python and separate from the physics: the game state is derived from the observed cube poses by snapping them to grid cells. Without an API key a clearly labelled, hand-coded offline heuristic (not an LLM) runs instead. A symbolic evaluator compares agents with a random baseline. Course: Physical AI 2026, final project.

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

## 実行

### デモ (物理シミュレーション + 動画)

```bash
./run.sh --offline --video outputs/demo_offline.mp4   # キー無し: オフラインのヒューリスティック (LLM ではない)
./run.sh --agent llm-hypo --cap 30 --speed 4 --video outputs/demo.mp4   # キーあり: Gemini (--agent の既定は llm-hypo)
./run.sh --offline --no-render                        # 動画なしでロジックだけ確認
```

`run.sh` (`pai.run`) のオプション: `--video PATH`, `--agent {llm-plain,llm-tricks,llm-hypo}` (既定 llm-hypo), `--offline`, `--no-render`, `--speed N` (動画の再生倍率。既定 4。画面に `Nx speed` と表示), `--cap N` (レベルあたりの行動数上限。既定 40), `--replay agent_log.jsonl` (過去の実行で実際に行われた判断を再生して動画を作り直す。API は呼ばない)。
キーが無い、または `--offline` / `PAI_OFFLINE` が指定されたときは手書きのオフライン・ヒューリスティックが動き、動画の HUD に `agent: offline heuristic` と表示されます。

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
report/build.sh    # 画像を用意し、Chrome headless で report/説明資料_PAI最終課題_Kakinotane.pdf を生成
```

`ffmpeg` と Google Chrome が必要です。`outputs/frames/*.png` があればそれを使い、無ければ `report/build.sh` 先頭の `VIDEO` と時刻の設定で動画から切り出します。

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

オフライン・ヒューリスティックは、規則の族と探索順序を人間が手で書き込んだ参照値であり、**LLM ではありません**。LLM 条件はエピソード数が少なく、条件間の優劣は結論できません。物理デモ (`outputs/demo.mp4`, llm-hypo) では L1/L2/L3 を 6/6/20 行動で解きました。

## 構成

```
src/pai/grid.py      盤面座標と ARC パレット (セル <-> ワールド座標, 姿勢 -> セル・段数)
src/pai/game.py      ゲームロジック (純 Python): 3 レベルの隠れルール, フィードバック, ランプ, 行動の検証, 履歴
src/pai/agents.py    random / オフラインヒューリスティック / Gemini の 3 条件 (plain, tricks, hypo) / ログ再生
src/pai/hypo.py      llm-hypo: 仮説プログラムの実行・反証・実験選択 (遷移シミュレータのみ。ランプの正解は呼ばない)
src/pai/scene.py     Genesis シーン (タイルとランプは見た目のみ) と Franka 制御 (IK + 摩擦把持)
src/pai/run.py       物理実行 + 動画収録 (姿勢を観測 -> ルール判定 -> 表示)
src/pai/evaluate.py  物理なしの評価 (random / heuristic / llm-plain / llm-tricks / llm-hypo)
src/pai/overlay.py   ARC 風 HUD・タイトル・LEVEL COMPLETE・結果カード
scripts/final_run.sh 最終実行 (デモ動画, フレーム, 3 条件の評価)
run.sh / eval.sh     実行ラッパ (.venv を有効化し PYTHONPATH=src で呼ぶ)
report/              説明資料: report.html -> PDF, build.sh, img/
outputs/             動画と結果 (上記)
```

## 設計メモ

物理とゲームロジックは分離しています。ロボットは行動を実行するだけで、実行後にキューブの姿勢を読み、セルにスナップした状態を `game.py` に渡します。ランプやタイルの表示は `game.py` の返り値をそのまま描いたものです。
LLM の出力は JSON で受け、行動を盤面の状態に対して検証し、不正なら理由を返して最大 3 回まで再生成させます (llm-hypo では仮説コードも制限付き実行で検証します)。無料枠の 429 / 5xx / タイムアウトは指数バックオフで再試行します。
API キーはコードに含めず、環境変数か `.env` から読みます。LLM の応答は実行ごとに変わり得ますが、物理シミュレーションは決定的です。
