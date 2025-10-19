# Kaggle Template (WSL + uv)

WSL2 上で最短に再現・実験できる Kaggle 用テンプレ。  
**WSL2 (Ubuntu) / Python 3.11 / uv + venv / pre-commit / PyTorch GPU**。

## Getting Started

```bash
# リポジトリを取得
git clone https://github.com/<your-name>/kaggle-template.git
cd kaggle-template
```

## Quick Start

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv が入る ~/.local/bin を PATH に追加（次回以降も使えるように .bashrc に書き込む）
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# --- Python 3.11 が未インストールの場合 ---
# Ubuntu 標準リポジトリにあれば:
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
# --------------------------------------------

# 仮想環境の作成と有効化（プロジェクト専用のPython環境）
python3.11 -m venv .venv
source .venv/bin/activate

# 依存関係を固定してインストール
uv lock
uv sync

# 開発・解析に必要なツールをインストール
uv add notebook jupyterlab ipykernel pre-commit japanize-matplotlib matplotlib-venn holiday holidays

# Git コミット前にコードを自動チェック/整形する仕組みを導入
#pre-commit install

# Jupyter Notebook/Lab からこの環境を選べるように登録（必要な場合だけ）
python -m ipykernel install --user --name kaggle-template --display-name "Python (kaggle-template)"
```

## Resume Work
```bash
# プロジェクトフォルダに移動
cd ~/kaggle-template

# 仮想環境を有効化
source .venv/bin/activate
```

## Project Layout

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
├─ data/        # Git管理外 (大規模データを置く)
├─ models/      # Git管理外 (学習済みモデルを置く)
└─ outputs/     # Git管理外 (予測結果や中間生成物を置く)
    ├─ oof/
    └─ preds/
```

## GPU (RTX3060)

### インストール方法

```bash
# PyTorch (GPU版) のインストール
# cu121 = CUDA 12.1 対応ビルド。自分のドライバに合ったものを選ぶ
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

# インストール確認 (バージョン / CUDA / GPU 利用可否を出力)
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

### 注意点

- Windows 側 NVIDIA Driver を最新にしておく
- WSL 用 CUDA Toolkit は不要、PyTorch の **cuXXX ホイール**を入れるだけでOK
- cu121 は CUDA 12.1 用。ドライバによっては cu118 や cu124 が必要になるので確認すること

## Tips
- 依存を追加：`uv add <pkg>` → `uv lock` → `uv sync`
- 依存を更新：`uv lock --upgrade-package <pkg>` → `uv sync`
- 解析前に `pre-commit run -a` で整形/静的解析

## License

MIT

