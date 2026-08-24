# MOKUMOKU Prototype

雲遊びゲーム「雲とひとびと / MOKUMOKU Prototype」の段階的プロトタイプです。

現在の実装範囲は Prototype A13 Cloud Surface Cohesion です。A0〜A12 基盤に加えて、内部ノードを描画下地へ吸収し、画面上へ見せる雲房を表面Lobe中心に間引くことで、Seed Ecologyで増えたノードが粒々のまま露出しにくい描画へ移行しています。

Quiet Motion Pass では、Ambient時のノード中心移動とサイズpulseを止め、外縁ノードだけを疎にsprite variant morphさせます。タップ、長押し、ドラッグ開始、フリックの反応は一回性のGrowthPulseとして扱い、グラフ距離2まで遅延伝播します。

Cloud Cohesion Polish では、接続タップ時の初期距離を短くし、通常時は内部パーツで隙間を埋め、伸長時は細いSTRETCHパーツへ切り替えて、接続ノードが丸と線ではなく一つの雲塊として読めるように調整しています。

Touch Response Pass では、タップ、ドラッグ開始、ドラッグ保持、長押し、リリースを別々の入力反応として扱います。Ambientの揺れは増やさず、入力時だけ局所的な成長・伸び・凝縮・戻りの描画を発火します。

Maturation / Settlement では、成熟してまとまった接続ノードの減衰をさらに抑え、弱い孤立断片や保持度の低い葉ノードは自然に薄くなりやすくしています。放置時はノイズが落ち、settledな雲ほど静かに残ります。

Prototype A5 Acceptance では、5〜8ノードの接続雲、全カメラでのタップ位置整合、30秒放置時のAmbient中心静止、通常表示のエッジ非表示を自動検査する受け入れシナリオを追加しています。

Cloud Reaction System では、短い押下は小さく集中した反応、長い押下は広く弱い放射反応として扱います。新規Seed生成は常に基本5個行い、押下時間に応じて最大9個まで増えます。疎な空では反応円内へSeedを複数配置し、中心Seedの後に周辺Seedが数フレームずつ遅れて現れます。周辺Seedは画面上の円形反応を保ったまま、world空間の縦長3D Birth Volume内で前後にも分散します。混在領域や密な雲ではSeed生成に加えて共鳴、既存雲の成長、グラフ伝播を重ねます。放射状に作られるSeedは中心に近いほどやや大きく、外側ほど軽く小さく生成されます。反応には生成数、反応ノード数、発芽数、共鳴数、イベント数、継続時間の予算を持たせ、終了後はQuiet Motionへ戻ります。

Cloud Seed Ecology では、弱いWaveを受けたDormant Seedはすぐ発芽せず `excitation` を蓄積します。2回目以降のWave、近くの発芽、周辺の空き具合によって閾値を超えた場合だけBloomし、Bloom後は短い不応期へ入ります。不応期中も薄いPulseは返しますが、同じSeedが連打で即座に増殖し続けることはありません。

Continuous Camera Dial では、地上下部の横型ダイヤルで yaw を -32°〜+32° の範囲で連続操作できます。ダイヤル操作中は雲入力を止め、生成済み雲のworld座標は変えずに投影と描画順だけを更新します。

Cloud Volume Readability Pass では、近接depthをバケット化して描画順の細かな反転を抑え、投影上の `INTERNAL` / `EDGE` / `BOTTOM` / `UPDRAFT` などの役割が境界で即座にパタつかないよう、短い保持時間とスコア差によるヒステリシスを入れています。さらにクラスタ内の相対depthから手前・中間・奥のlayerを割り当て、奥の雲房は少し弱い色で描いて立体雲の重なりを読みやすくしています。

Stacked Birth Volume では、既存雲の上端や上昇流の強いノードを刺激した時、新規Seed候補のBirth Volumeを小さめにして上方向へ寄せます。これにより、同じ反応円でも雲房が水平に広がるだけでなく、上へ積み上がる入道雲らしい成長を始めます。

Upper Cloud Attachment では、積み上げ反応で生まれた中心〜中間Seedの一部を刺激元ノードへ接続し、独立した点ではなく既存雲の上に乗った雲房として育つようにしています。外縁Seedは次の共鳴・発芽の余地として残します。

Directional Cloud Growth では、親Seedの `polarity`、クラスタ外向き、上昇流、周囲12方向の空き具合を使ってSecondary Sproutの候補を評価します。子Seedは親の方向性と実際の成長方向を混ぜて `polarity` を継承するため、雲房ごとに縦へ伸びる、横へ広がる、外へ逃げるといった差が残ります。

Growth Shape Polish では、親ノードの周囲12方向を見て外縁度を算出します。内部に埋まったノードは発芽数と候補スコアが下がり、空いている外縁やクラスタ外向きの方向ほど子Seedを出しやすくなります。

Seed Ecology Phase 2 では、Primed Seedが近くに複数ある場合に弱いSeedを後押しするクオラム信号を追加しています。一方で中距離の密な雲房は発芽閾値と候補スコアを押し下げるため、全部が一斉に膨らむのではなく、空いた場所で遅れてBloomしやすくなります。

Cloud Surface Cohesion では、CloudNodeをそのまま全て可視スプライト化せず、同じlineage内の投影近傍から露出率を計算します。密な内部ノードは輪郭付きLobeを抑え、Body PassとGap Fillで白い雲塊の下地へ吸収します。表面候補は露出率、見かけ半径、上昇流、安定度でスコア化し、近すぎる候補をNon-Maximum Suppressionで間引きます。小さなDormant Seedと反応中ノードは従来どおり見えるため、生成直後の粒感と成熟後の雲塊感を分けています。

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

- `Q`: 左側の基準カメラへ
- `E`: 右側の基準カメラへ
- `C`: カメラ循環
- 地上の `<` / `>`: 左右カメラへ
- 地上下部のダイヤル: カメラyawを連続変更
- ダイヤル中央目盛り: 正面0°へ戻す
- 空で短く押して離す: 基本5個のDormant Seedを放射状に作る
- 空で長く押して離す: 押下時間に応じて最大9個のDormant Seedを放射状に作る
- 雲で押して離す: 周辺密度に応じて刺激、遅延伝播、副次発芽
- 近くのDormant Seed: 反応波から距離に応じて遅れて共鳴
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
