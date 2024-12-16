# tuning-competition-baseline

## 概要
このリポジトリは，言語処理学会第31回年次大会 (NLP2025) のワークショップ「[大規模言語モデルのファインチューニング技術と評価](https://llm-jp.github.io/tuning-competition/)」で開催されるコンペティションのベースラインモデルを作成するコードです．

このコードは [Nemo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) をベースにしています．
同様の機能を提供している [trl](https://github.com/huggingface/trl) と比較して，大規模な計算資源を用意できる場合には計算効率が良いです．そのため計算資源が少ない場合などは trl の方が向いていることがあります．trl によるインストラクションチューニングは [llm-jp-sft](https://github.com/llm-jp/llm-jp-sft) で行えます．
今回のコンペティション用には整備されていないですが，必要に応じて参考にしてください．

## インストール

### 環境構築
```bash
cd /path/to/your/project_dir # 任意のディレクトリ
git clone https://github.com/llm-jp/tuning-competition-baseline.git
cd tuning-competition-baseline
bash ./scripts/mdx/install.sh
```
注意：上記の環境構築スクリプトは `CUDA Toolkit: 11.8` 用に設定されています．他のバージョンのCUDA Toolkitを使用する場合は，`scripts/mdx/install.sh` や `requirements.txt` を適宜修正してください．


### チューニング用データセットのダウンロード

再配布可能なライセンスを持つデータセットは[こちら](https://drive.google.com/drive/folders/1sPURwuXSjDS_hrnJmqMLfYJsu_fCwYel)に前処理済みのものを準備しているのでダウンロードしてください．
ダウンロードした zip ファイルは解凍すると以下のような構造になっています．
```
tuning-competition-datasets/
├── calm3_22b_chat_20241018083433--Qwen2.5_32B_Instruct_20241022115410.jsonl
├── calm3_22b_chat_20241022133932--Qwen2.5_32B_Instruct_20241024100350.jsonl
├── calm3_22b_chat_20241022155627--Qwen2.5_32B_Instruct_20241024144245.jsonl
├── daring_anteater_en.jsonl
├── flan.jsonl
├── logical_math_coding_wizard8x22b.jsonl
├── multiturn_calm3.jsonl
├── random_to_fixed_multiturn_calm3.jsonl
├── synthetic_jp_en_coding_0.jsonl
```

この `tuning-competition-datasets/` を `datasets/` に移動してください．

これらのデータセットの他に [LLMのための日本語インストラクションデータ](https://liat-aip.sakura.ne.jp/wp/llm%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%A9%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%87%E3%83%BC%E3%82%BF%E4%BD%9C%E6%88%90/llm%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%A9%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%87%E3%83%BC%E3%82%BF-%E5%85%AC%E9%96%8B/) と [AnswerCarefully Dataset](https://liat-aip.sakura.ne.jp/wp/answercarefully-dataset/) を使用します．
これらのデータセットは再配布が許可されていないため，各自で利用規約を確認したうえでダウンロードし，解凍して `datasets/` に配置してください．
これらのデータセットの前処理は以下のコマンドで行います．
```bash
python preprocess.py \
  --ichikara-dir ./datasets/tuning-competition-datasets/Distribution20241221_all \
  --answer-carefully-dir ./datasets/tuning-competition-datasets/AnswerCarefullyVersion002 \
  --output-dir ./datasets/tuning-competition-datasets
```

### 設定ファイルの作成
```bash
cp configs/base_template.yaml configs/base.yaml
```
`configs/base.yaml` に実験の基本的な設定を記述します．テンプレートファイルをコピーして `base.yaml` を作成し，`FIXME` と記載されている箇所を修正してください．

### wandb の設定
このコードでは実験ログの管理に [wandb](https://wandb.ai/site/ja/) をデフォルトで使用しています．
wandb のアカウントを作成した上で `wandb login` を実行してください．
詳細は [wandb のドキュメント](https://docs.wandb.ai/ja/quickstart/) を参照してください．

wandb を使用しない場合は `configs/exp_manager/sft.yaml` の 4行目 `create_wandb_logger` を `False` に変更してください．

## 学習

### チェックポイント変換（Hugging Face -> Nemo）

Nemo-Aligner では Hugging Face のモデルを直接使用することはできず，Nemo フォーマットへの変換が必要です．
`FIXME` と記載されている箇所を修正した上で，以下のスクリプトを実行してください．

```bash
bash ./scripts/mdx/ckpt/convert_llama_hf_to_nemo.sh
```

### 学習の実行
`FIXME` と記載されている箇所を修正した上で，以下のスクリプトを実行してください．

```bash
bash ./scripts/mdx/mpi/train.sh
```
学習スクリプト (`./scripts/mdx/mpi/train.sh`) の 23行目に記載されている `${NAME}` は学習するモデルの名前で毎回ランダムな文字列が生成され，学習結果を保存するディレクトリ名としても使用されます．
たとえば `sft-LN0KRzXR0V` などです．

### チェックポイント変換（Nemo -> Hugging Face）
Nemo フォーマットのままでも推論を行うことはできますが，よく使われる評価用コードのほとんどが Hugging Face のモデルを前提としているため，Nemo フォーマットから Hugging Face フォーマットへの変換を行います．
`FIXME` と記載されている箇所を修正した上で，以下のスクリプトを実行してください．
${NAME} には学習時に生成されたモデルの名前を指定してください（例：`sft-LN0KRzXR0V`）．

```bash
bash ./scripts/mdx/ckpt/convert_llama_nemo_to_hf.sh ${NAME}
```