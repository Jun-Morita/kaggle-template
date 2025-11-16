# Kaggle Template (WSL2 + uv)

WSL2 上で**最短**に再現・実験できる Kaggle 用テンプレ。  
**WSL2 (Ubuntu)** / **Python 3.11** / **uv + venv** / **pre-commit** / **PyTorch GPU** 対応。

---

## 1. 前提

- Windows 10/11 + **WSL2** 有効化済み（Ubuntu 推奨）
- Python 3.11 を利用（無ければ後述の手順で導入）
- NVIDIA GPU を使う場合は **Windows 側ドライバを最新化**  
  ※ WSL 内に CUDA Toolkit の個別インストールは不要（PyTorch の *cuXXX* ホイールでOK）

---

## 2. リポジトリ取得（GitHub と連携したい場合）

```bash
git clone https://github.com/Jun-Morita/kaggle-template.git
cd kaggle-template
````

---

## 2.1 Git管理せずにテンプレだけ使いたい場合

テンプレの構成だけ欲しく、**このプロジェクトで Git を使わない場合**はこちら：

```bash
git clone https://github.com/Jun-Morita/kaggle-template.git
cd kaggle-template

# Git履歴を削除して通常フォルダ化
rm -rf .git
```

これで Git 管理なしの “生のテンプレ” として利用できます。
構成や設定はそのまま使えるため、個人用・単発プロジェクトに最適です。

---

## 3. セットアップ（初回のみ）

### 3.1 uv の導入 + PATH 追記

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3.2 Python 3.11 をまだ入れていない場合

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### 3.3 仮想環境の作成と依存インストール

```bash
python3.11 -m venv .venv
source .venv/bin/activate
uv lock && uv sync
uv add notebook jupyterlab ipykernel pre-commit japanize-matplotlib matplotlib-venn holidays
pre-commit install
```

> `uv add` でよく使うパッケージをまとめて入れています。  
> 必要に応じて追加・削除してください。

### 3.4 Jupyter カーネル登録（必要な場合のみ）

```bash
python -m ipykernel install --user --name kaggle-template --display-name "Python (kaggle-template)"
```

---

## 4. ふだんの使い方（Resume Work）

```bash
cd ~/kaggle-template   # または自分のプロジェクトフォルダ
source .venv/bin/activate
```

---

## 5. Git clone 版 vs Template Only 版 比較

| 目的                             | 方法                                     | Git管理 | 説明         |
| ------------------------------ | -------------------------------------- | ----- | ---------- |
| Kaggle用テンプレをベースに開発したい          | ✅ Git clone 版                          | 継続    | 最も推奨。再現性高い |
| テンプレだけ取得し、このプロジェクトではGitを使いたくない | ✅ Clone → `rm -rf .git`（Template Only） | なし    | 個人用・単発向け   |

---

## 6. pyproject.toml あり/なし の違い

| 運用スタイル                     | 特徴                        | どんな人向け？          |
| -------------------------- | ------------------------- | ---------------- |
| **pyproject.toml あり（推奨）**  | 依存が明確。再現性高い。環境差異が起きにくい。   | 本格運用、複数PC、チーム    |
| **pyproject.toml なし（超手軽）** | `uv add` で入れた分の依存だけ保持。軽量。 | 個人学習、試行用、雑に始めたい時 |

🧠 **迷ったら “あり” が正解**
再現性とバージョン管理の観点でメリットが圧倒的。

---

## 7. プロジェクト構成

```
kaggle-template/
├─ pyproject.toml
├─ uv.lock
├─ .pre-commit-config.yaml
├─ .gitignore
├─ README.md
├─ src/
├─ notebooks/
│   └─ 00_eda.ipynb
├─ data/        # Git管理外（大規模データ）
├─ models/      # Git管理外（学習済みモデル）
└─ outputs/     # Git管理外（予測/中間生成物）
    ├─ oof/
    └─ preds/
```

---

## 8. GPU (RTX3060 など)

### インストール

```bash
# 例: CUDA 12.1 対応 (cu121) の PyTorch を入れる
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
```

> **メモ**: ドライバや環境に応じて
>
> * CUDA 11.8: `.../cu118`
> * CUDA 12.4: `.../cu124`
>   などに差し替えてください。
>   CPU 版にしたい場合は extra-index-url を付けずに：
>   `uv add torch torchvision torchaudio`

### 動作確認

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name())
PY
```

---

## 9. Tips

* 依存追加：`uv add <pkg>` → `uv lock` → `uv sync`
* 依存更新：`uv lock --upgrade-package <pkg>` → `uv sync`
* 解析前の整形/静的解析：`pre-commit run -a`

---

## 10. よくある詰まりポイント

* **`uv: command not found`**
  → `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc`
* **Jupyter にカーネルが出ない**
  → `.venv` を有効化後に `python -m ipykernel install --user --name kaggle-template`
* **GPU が使えない**
  → Windows 側 NVIDIA ドライバ更新／PyTorch の *cuXXX* を環境に合わせて再インストール
