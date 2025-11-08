# Kaggle Template (WSL2 + uv)

WSL2 上で**最短**に再現・実験できる Kaggle 用テンプレ。  
**WSL2 (Ubuntu)** / **Python 3.11** / **uv + venv** / **pre-commit** / **PyTorch GPU** 対応。

---

## 0. 前提
- Windows 10/11 + **WSL2** 有効化済み（Ubuntu 推奨）
- Python 3.11 を利用（無ければ後述の手順で導入）
- NVIDIA GPU を使う場合は **Windows 側ドライバを最新化**  
  ※ WSL 内に CUDA Toolkit の個別インストールは不要（PyTorch の *cuXXX* ホイールでOK）

---

## 1. リポジトリ取得

```bash
git clone https://github.com/Jun-Morita/kaggle-template.git
cd kaggle-template
````

---

## 2. セットアップ（初回のみ）

```bash
# uv のインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# ~/.local/bin を PATH に追加（永続化）
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# ---- Python 3.11 が未インストールの場合 ----
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
# ---------------------------------------------

# プロジェクト専用の仮想環境を作成・有効化
python3.11 -m venv .venv
source .venv/bin/activate

# 依存関係をロック＆インストール
uv lock
uv sync

# 開発ツール・よく使う解析用パッケージ（必要に応じて調整）
uv add notebook jupyterlab ipykernel pre-commit japanize-matplotlib matplotlib-venn holidays
#  ↑ 注: 'holiday' ではなく 'holidays'（python-holidays）です

# pre-commit（コミット前の自動整形・静的チェック）を有効化
pre-commit install

# Jupyter からこの環境を選べるようカーネル登録（必要な場合のみ）
python -m ipykernel install --user --name kaggle-template --display-name "Python (kaggle-template)"
```

---

## 3. ふだんの使い方（Resume Work）

```bash
cd ~/kaggle-template
source .venv/bin/activate
```

---

## 4. プロジェクト構成

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

## 5. GPU (RTX3060 など)

### インストール

```bash
# 例: CUDA 12.1 対応 (cu121) の PyTorch を入れる
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

> **メモ**: ドライバや環境に応じて
>
> * CUDA 11.8: `.../cu118`
> * CUDA 12.4: `.../cu124`
>   などに差し替えてください。CPU 版にしたい場合は extra-index-url を付けずに `uv add torch torchvision torchaudio` でOK。

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

## 6. Tips

* 依存追加：`uv add <pkg>` → `uv lock` → `uv sync`
* 依存更新：`uv lock --upgrade-package <pkg>` → `uv sync`
* 解析前の整形/静的解析：`pre-commit run -a`

---

## 7. よくある詰まりポイント

* **`uv: command not found`**
  → `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc`
* **Jupyter にカーネルが出ない**
  → `.venv` を有効化後に `python -m ipykernel install --user --name kaggle-template`
* **GPU が使えない**
  → Windows 側 NVIDIA ドライバ更新／PyTorch の *cuXXX* を環境に合わせて再インストール
