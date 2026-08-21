# MOKUMOKU Prototype

雲遊びゲーム「雲とひとびと / MOKUMOKU Prototype」の段階的プロトタイプです。

現在の実装範囲は Prototype A / A4 です。A0〜A3 基盤に加えて、`assets/mokumoku.pyxres` の雲パーツ読み込み、スプライトマップ、深度順描画、Web 向け touch 設定を実装しています。

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
- 空をタップ: 最初の雲の種を作る
- 雲をタップ: 局所成長
- 雲を長押し: 局所凝縮
- 雲をドラッグ: 観察平面上で変形
- 雲をフリック: 断片へ分裂
- 分裂した断片を近づける: 合流
- `D`: デバッグ表示切り替え
- `F4`: 熟成を 8 秒進める
- `Esc`: 終了

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

成果物は `docs/index.html` に出力します。GitHub Pages では `docs/` を公開対象にします。

確認だけなら Pyxel CLI なしで dry-run できます。

```bash
python3 scripts/build_web.py --dry-run
```
