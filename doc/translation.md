# 翻訳

## 実行方法

### ソースディレクトリ

(rood dir)/src/translation

### 学習データ生成

0. 以下のリポジトリをクローンする。
https://github.com/odashi/small_parallel_enja
0. 以下のコマンドを実行する。(dataset_repository_rootはクローンしたデータセットリポジトリのルートディレクトリ)。
```
python make_dataset.py dataset/dataset.json dataset/dataset.pkl -d (dataset_repository_root)
```

### 学習

```
python train.py [dataset file] [input language] [output languate] [output model path] [output result path]
```

例:

```
$ cd src/translation
$ python train.py dataset/dataset.pkl en ja model/en_ja.model model/en_ja_result.txt
```

パラメータ:
* `dataset file` : データセットファイルのパス
* `input language` : 入力言語
* `output language` : 出力言語
* `output model path` : 出力モデルファイルパス
* `output result path` : テストデータの翻訳結果の出力ファイルパス

オプション:
* `-e <epoch>` : 学習epoch数(default: 50)
* `-b <batch size>` : ミニバッチサイズ(default: 100)
* `--hidden-size <size>` : 隠れ層のサイズ(default: 256)
* `--max-result-len <length>` : テストデータ翻訳結果の最大単語数
