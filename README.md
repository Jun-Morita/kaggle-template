# Kaggle Template (WSL + uv)

WSL2 上で最短に再現・実験できる Kaggle 用テンプレ。  
**WSL2 (Ubuntu) / Python 3.11 / uv + venv / pre-commit / PyTorch GPU**。

## Quick Start

```bash
# uv & venv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

python3.11 -m venv .venv
source .venv/bin/activate

# 依存ロック & 同期
uv lock
uv sync

# 品質ツール（初回）
pre-commit install

# Jupyter カーネル（必要なら）
python -m ipykernel install --user --name kaggle-template
```

## Project Layout

```
kaggle-template/
├─ pyproject.toml
├─ uv.lock
├─ .pre-commit-config.yaml
├─ .gitignore  (任意)
├─ README.md
├─ src/
├─ notebooks/
│   └─ 00_eda.ipynb
├─ data/
├─ models/
└─ outputs/
    ├─ oof/
    └─ preds/
```

## GPU (RTX3060)

- Windows 側 NVIDIA Driver を最新、WSL 用 CUDA は PyTorch の cuXXX ホイールで賄う方針（Toolkitは不要）

```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'is_available:', torch.cuda.is_available())
PY
```

## Tips
- 依存追加：`uv add <pkg>` → `uv lock` → `uv sync`
- 解析前に `pre-commit run -a` で整形/静的解析

## License

MIT

