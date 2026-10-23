# LLM × Genesis: 自然言語で指示するロボットアーム (Physical AI講座 2026 応用編1 最終課題)

自然言語(日本語/English)の指示を LLM が JSON プランに変換し、Genesis 上の Franka Panda が実行する。
各動作の後にシミュレータの物体姿勢から成否を観測し、失敗・外乱があれば **観測状態と失敗理由を LLM に返して再計画** する。

*English summary:* a Franka Panda in the Genesis simulator executes tabletop pick-and-place tasks given as natural-language
instructions. An LLM turns the instruction into a JSON plan of `pick` / `place_on` primitives; the plan is symbolically validated
(invalid plans are bounced back to the LLM with the reason). After each step the simulator state is observed, and on failure or
disturbance the observed state is fed back to the LLM for re-planning.

## 構成

```
src/pai/scene.py     Genesis シーン, Franka 制御 (IK + PD, 摩擦把持, 手内姿勢補償)
src/pai/planner.py   LLM プランナー (Azure OpenAI / OpenAI / オフライン), JSON 検証, 再計画
src/pai/run.py       閉ループ実行 (観測 → 成否判定 → 再計画), 動画収録
src/pai/overlay.py   字幕・プラン表示 (Pillow)
run.sh               実行ラッパ
report/              説明資料 (report.html → PDF), build.sh
outputs/demo.mp4     デモ動画
```

## セットアップ (macOS / Linux, CPU でOK)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env     # LLM を使う場合のみ。キーを記入 (.env は git 管理外)
```

## 実行

```bash
./run.sh --demo                                   # outputs/demo.mp4 を生成 (2シナリオ, 約80秒, CPUで1〜2分)
./run.sh -i "赤いキューブを青い皿の上に置いて、そのあと緑のキューブを赤の上に積んで" --disturb --video out.mp4
./run.sh -i "Put the red cube on the blue plate." --no-render   # 動画なしでロジックのみ確認
```

`--disturb` は 1 動作目の後にキューブを払い落とす外乱を人為的に注入し、失敗検知 → 再計画を再現する。

### LLM バックエンド

`.env` に `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_DEPLOYMENT_NAME` があれば Azure OpenAI、
なければ `OPENAI_API_KEY` があれば OpenAI を使う。どちらも無い場合 (または `PAI_OFFLINE=1`) は
`plan_cache.json` (過去の LLM 応答; LLM 実行時に自動作成) → 小さなルールベース解析器の順にフォールバックする。
どのプランナーが使われたかは動画下部の `planner:` 表示に出る。

オブジェクト: `red_cube`, `green_cube`, `blue_cube`, `yellow_cube` (掴める), `blue_plate` (固定の皿)。
プリミティブ: `{"action":"pick","object":ID}` / `{"action":"place_on","target":ID}`。

## 説明資料

```bash
report/build.sh    # demo.mp4 からスクリーンショットを切り出し、Chrome headless で PDF 化
```

## 注意

APIキーはコードに含めず環境変数 / `.env` から読む。シミュレーションは決定的だが、LLM の応答は実行ごとに変わりうる。
