# MOKUMOKU Prototype

雲遊びゲーム「雲とひとびと / MOKUMOKU Prototype」の段階的プロトタイプです。

現在の実装範囲は Prototype A6 Cloud Reaction です。A0〜A5 基盤に加えて、既存雲へのタップを単純な1ノード追加ではなく CloudStimulus として処理し、Neighbor Pulse、Secondary Sprout、Seed Resonance を一回性イベントとして実装しています。

Quiet Motion Pass では、Ambient時のノード中心移動とサイズpulseを止め、外縁ノードだけを疎にsprite variant morphさせます。タップ、長押し、ドラッグ開始、フリックの反応は一回性のGrowthPulseとして扱い、グラフ距離2まで遅延伝播します。

Cloud Cohesion Polish では、接続タップ時の初期距離を短くし、通常時は内部パーツで隙間を埋め、伸長時は細いSTRETCHパーツへ切り替えて、接続ノードが丸と線ではなく一つの雲塊として読めるように調整しています。

Touch Response Pass では、タップ、ドラッグ開始、ドラッグ保持、長押し、リリースを別々の入力反応として扱います。Ambientの揺れは増やさず、入力時だけ局所的な成長・伸び・凝縮・戻りの描画を発火します。

Maturation / Settlement では、成熟してまとまった接続ノードの減衰をさらに抑え、弱い孤立断片や保持度の低い葉ノードは自然に薄くなりやすくしています。放置時はノイズが落ち、settledな雲ほど静かに残ります。

Prototype A5 Acceptance では、5〜8ノードの接続雲、全カメラでのタップ位置整合、30秒放置時のAmbient中心静止、通常表示のエッジ非表示を自動検査する受け入れシナリオを追加しています。

Cloud Reaction System では、完全な空白タップだけDormant Seedを1個作り、既存雲や近くの種へのタップは刺激として扱います。触れた雲房が即時反応し、隣接ノードへ数フレーム遅れて伝播し、条件に応じて子ノードが副次発芽し、周辺のDormant Seedが距離に応じて遅れて共鳴します。反応には世代数、反応ノード数、発芽数、共鳴数、イベント数の予算を持たせ、終了後はQuiet Motionへ戻ります。

## Requirements

- Python 3.11+
- Pyxel 2.9.9

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

## Run

```bash
python main.py --seed 12345
```

GUI なしの起動スモーク:

```bash
python main.py --headless --smoke-frames 5
```

- `Q`: 左カメラへ
- `E`: 右カメラへ
- `C`: カメラ循環
- 地上の `<` / `>`: 左右カメラへ
- 完全な空をタップ: Dormant Seedを作る
- 雲をタップ: 刺激、遅延伝播、副次発芽
- 近くのDormant Seed: 刺激や周辺雲から遅れて共鳴
- 雲を長押し: 局所凝縮
- 雲をドラッグ: 観察平面上で変形
- 雲をフリック: 断片へ分裂
- 分裂した断片を近づける: 合流
- `D`: デバッグ表示切り替え
- `F4`: 熟成を 8 秒進める
- `Esc`: 終了

通常表示では雲の接続線を描かず、デバッグ表示中のみエッジ線、内部状態、平均保持度を表示します。

## Test

```bash
python3 -m unittest discover -s tests
```

Prototype A 受け入れシナリオだけを確認する場合:

```bash
python3 scripts/check_prototype_a_acceptance.py
```

## Web Build

Pyxel CLI が使える環境で実行します。

```bash
python3 scripts/generate_mokumoku_resource.py
python3 scripts/build_web.py
```

成果物は `docs/index.html` と `docs/builds/<build_id>/index.html` に出力します。
GitHub Pages では `docs/` を公開対象にします。

`build_id` は `.pyxapp` の内容から作る12桁のハッシュです。実機確認では
`https://sejiseji.github.io/mokumoku/builds/<build_id>/` を使うと、クエリ文字列だけでなく
HTMLのパスとPyxel起動名も変わるため、Safariのキャッシュに左右されにくくなります。
固定版ビルドは最新3件だけを保持し、3つ前以降の `docs/builds/` は次回ビルド時に削除します。

確認だけなら Pyxel CLI なしで dry-run できます。

```bash
python3 scripts/build_web.py --dry-run
```
