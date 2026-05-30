# Kaggle Template (WSL2 + uv)

WSL2 で Kaggle の検証をすぐ始めるためのテンプレートです。  
環境構築用の設定と、サンプルノートブック（Titanic / House Prices / Leaf Classification /
Bike Sharing Demand）を含みます。

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
# または
cd sample/leaf-classification
# または
cd sample/bike-sharing-demand
```

サンプルNotebookは公開テンプレートとして初回実行を軽くするため、`N_SEEDS = 1` にしています。  
計算時間に余裕があれば、`N_SEEDS` を増やして予測の分散を抑えられます。
`N_SPLIT` は単純に増やすのではなく、データ量と validation の設計に合わせて調整してください。

### 4.1 サンプルNotebookの選び方

4つのNotebookは、それぞれ単独で実行できます。最初は Titanic で全体の流れを確認し、
取り組みたいタスクに近いNotebookへ進むと理解しやすくなります。

| Notebook | タスク | 主な評価指標 | 検証方法 | 学ぶポイント |
| --- | --- | --- | --- | --- |
| Titanic | 二値分類 | ROC AUC | Stratified KFold | 欠損値、カテゴリ特徴量、確率予測 |
| House Prices | テーブル回帰 | RMSLE | KFold | 対数変換、特徴量作成、残差分析 |
| Leaf Classification | 多値分類 | multiclass log loss | Stratified KFold | クラス確率、混同行列、信頼度分析 |
| Bike Sharing Demand | 時系列回帰 | RMSLE | expanding-window CV | 日時特徴量、未来情報のリーク、期間別検証 |

各Notebookは、次の順序で構成しています。

1. データ読み込みと EDA
2. 特徴量作成
3. competition に合わせた validation
4. LightGBM / XGBoost / CatBoost / 線形モデルの学習
5. OOF 予測による評価と seed averaging
6. 推論と `submission.csv` の作成
7. feature importance と予測診断

学習済みモデル、OOF 予測、提出ファイルは、各サンプルディレクトリ配下の `models/` と
`oof/` に保存されます。これらは Git 管理外です。

欠損値の補完や one-hot encoding など、データから統計量や変換ルールを学習する処理は、
原則として各 fold の学習データだけで `fit` します。サンプルでは線形モデルの
`Pipeline` にこれらの処理を含め、validation の情報が学習へ混ざらないようにしています。
木モデルへ渡す `category` dtype のカテゴリ定義は train/test で表現を揃えるため共通化しますが、
目的変数や集計値は使用しません。

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
   ├─ houseprices/
   │  ├─ houseprices.ipynb
   │  └─ data/
   ├─ leaf-classification/
   │  ├─ leaf-classification.ipynb
   │  └─ data/
   └─ bike-sharing-demand/
      ├─ bike-sharing-demand.ipynb
      └─ data/
```

補足:
- `sample/*/models`, `sample/*/oof`, `sample/*/catboost_info` などの学習済み成果物は Git 管理外です。
- いまの実体は `sample/` 中心の構成です。

## 6. サンプルデータの出典

サンプルでは、以下の Kaggle Competition で配布されているデータを使用します。

- [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Leaf Classification](https://www.kaggle.com/competitions/leaf-classification)
- [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)

このリポジトリには、Notebook を単独で実行できるようにサンプルデータを同梱しています。
Competition 用に整形されたファイルを利用・再配布する場合は、原典の条件だけでなく、
各 Competition ページの Rules も確認してください。
リポジトリ本体の MIT License は、以下のデータには適用されません。
以下では、確認できた原典の citation と license / usage terms を記載します。
Competition 版では、train/test 分割、列名、対象クラスなどが原典から変更されている場合があります。

### 6.1 Titanic

Titanic の乗客データについては、Vanderbilt University Department of Biostatistics の
データセット解説を参照してください。解説では、主な情報源として Encyclopedia Titanica が
挙げられています。

Original dataset and provenance:
- [Titanic Data](https://hbiostat.org/data/repo/titanic.html)
- [Vanderbilt Biostatistics Datasets](https://hbiostat.org/data/)
- [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/)

License / usage terms:
- Vanderbilt Biostatistics Datasets では、提供データセットの利用を許可し、原典および
  Vanderbilt University Department of Biostatistics から取得した旨の記載を求めています。
- 標準化されたオープンデータライセンスは明記されていません。

### 6.2 House Prices

House Prices のデータは、Ames City Assessor's Office の記録を基に Dean De Cock が整備した
Ames Housing Data を使用しています。

Citation:

> De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an
> End of Semester Regression Project. Journal of Statistics Education, 19(3).

Original dataset and provenance:
- [Ames Housing Data paper](https://jse.amstat.org/v19n3/decock.pdf)
- [Ames Housing data](https://jse.amstat.org/v19n3/decock/AmesHousing.txt)

License / usage terms:
- 原典論文では、Ames City Assessor's Office から受領した元データを共有する意向が
  示されています。
- 標準化されたオープンデータライセンスは明記されていません。

### 6.3 Leaf Classification

Leaf Classification のデータは、UCI Machine Learning Repository で公開されているデータセットを
基にしています。
このリポジトリでは、Kaggle Competition で配布されている 99 種・1,584 画像の版を使用します。

Citation:

> Cope, J., Beghin, T., Remagnino, P., & Barman, S. (2013).
> One-hundred plant species leaves data set [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5RG76.

Original dataset:
- [One-hundred plant species leaves data set](https://archive.ics.uci.edu/dataset/241/one+hundred+plant+species+leaves+data+set)

Original dataset license:
- [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

### 6.4 Bike Sharing Demand

Bike Sharing Demand のデータは、Capital Bikeshare の時間単位・日単位のレンタル数と
天候・季節情報をまとめた UCI Machine Learning Repository のデータセットを基にしています。

Citation:

> Fanaee-T, H. (2013). Bike Sharing [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

Original dataset:
- [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

Original dataset license:
- [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## 7. コミット前のチェック

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

## 8. GPU (PyTorch) を使う場合

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

## 9. よく使うコマンド

- 依存追加: `uv add <pkg>` → `uv lock` → `uv sync`
- 依存更新: `uv lock --upgrade-package <pkg>` → `uv sync`
- 静的解析・整形: `pre-commit run -a`
