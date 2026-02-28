# Kaggle Template (WSL2 + uv)

WSL2 で Kaggle の検証をすぐ始めるためのテンプレートです。  
環境構築用の設定と、サンプルノートブック（Titanic / House Prices）を含みます。

## 1. こんな人向け

- WSL2 で Python 環境を安定して運用したい
- Kaggle の作業を再現しやすい形で始めたい
- サンプルNotebookをベースに素早く検証したい

## 2. 前提

- Windows 10/11 + WSL2 (Ubuntu 推奨)
- Python 3.11
- `uv`
- GPU を使う場合は Windows 側 NVIDIA ドライバを最新化

## 3. 最短スタート

### 3.1 取得

```bash
git clone https://github.com/Jun-Morita/kaggle-template.git
cd kaggle-template
```

Git 管理せずにテンプレだけ使うなら:

```bash
rm -rf .git
```

### 3.2 初回セットアップ

`uv` が未導入なら:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Python 3.11 が未導入なら:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

仮想環境作成と依存同期:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
uv sync
pre-commit install
```

必要な場合のみ Jupyter カーネル登録:

```bash
python -m ipykernel install --user --name kaggle-template --display-name "Python (kaggle-template)"
```

## 4. 日常作業

```bash
cd ~/kaggle-template
source .venv/bin/activate
```

サンプルNotebookを使う場合は、対象ディレクトリに移動して実行してください。

```bash
cd sample/titanic
# または
cd sample/houseprices
```

## 5. リポジトリ構成（現在）

```text
kaggle-template/
├─ pyproject.toml
├─ uv.lock
├─ .pre-commit-config.yaml
├─ .gitignore
├─ README.md
└─ sample/
   ├─ titanic/
   │  ├─ titanic.ipynb
   │  └─ data/
   └─ houseprices/
      ├─ houseprices.ipynb
      └─ data/
```

補足:
- `sample/*/models`, `sample/*/oof`, `sample/*/catboost_info` には学習済み成果物が入っています。
- いまの実体は `sample/` 中心の構成です。

## 6. コミット前のチェック

コミット前に `ruff` を実行して、整形と lint を先に通す運用を推奨します。

推奨手順:

```bash
# 1) lint（自動修正）
uv run ruff check --fix .

# 2) format
uv run ruff format .

# 3) 変更確認
git status
```

`pre-commit install` 済みの場合は、`git commit` 時にも同様のチェックが実行されます。  
フックで修正が入った場合は、再度 `git add` してコミットしてください。

## 7. GPU (PyTorch) を使う場合

```bash
# 例: CUDA 12.1 (cu121)
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
```

確認:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
PY
```

## 8. よく使うコマンド

- 依存追加: `uv add <pkg>` → `uv lock` → `uv sync`
- 依存更新: `uv lock --upgrade-package <pkg>` → `uv sync`
- 静的解析・整形: `pre-commit run -a`
