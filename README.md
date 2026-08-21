# MOKUMOKU Prototype

雲遊びゲーム「雲とひとびと / MOKUMOKU Prototype」の段階的プロトタイプです。

現在の実装範囲は Prototype A4 Foundation + A4.5 Cloud Cohesion 初期パスです。A0〜A4 基盤に加えて、通常表示の接続線非表示、雲ブリッジ、投影ベースのスプライト役割判定、小さい単体雲のメッシュ風描画、スマホ用カメラボタン、まとまった雲ほど残りやすい熟成保持を実装しています。

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
- 空をタップ: 最初の雲の種を作る
- 雲をタップ: 局所成長
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

確認だけなら Pyxel CLI なしで dry-run できます。

```bash
python3 scripts/build_web.py --dry-run
```
