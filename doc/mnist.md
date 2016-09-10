# 手書き数字の認識

## MNISTデータセット

http://yann.lecun.com/exdb/mnist/

* 手書き数字の画像データセット
* 28 x 28px の白黒画像
* 学習データ60,000枚、テストデータ60,000枚

## 実行方法

### ソースディレクトリ

(root dir)/src/mnist

### 学習

```
$ cd src/mnist
$ python train.py
```

オプション:
* `-m <model name>` : ニューラルネットワークのモデルを指定する(default: mlp)
    * mlp: Fully Connected Layerのみ
    * cnn: Convolutional Neural Networkを使用
* `-g <GPU index>` : 使用するGPUを指定する。-1の場合はCPUを使用(default: -1)
* `-e <epoch>` : epoch数を指定する(default: 100)
* `-b <batch size>` : ミニバッチ数を指定する
* `-p <prefix>` : モデルパラメータ保存時のファイル名のprefixを指定する

### 予測

```
$ python predict.py mnist.model sample.png
```

パラメータ:
* 学習済みモデルパラメータファイル
* 画像ファイルパス。画像は28 x 28px にリサイズされる

オプション:
* `-m <model name>` : ニューラルネットワークのモデルを指定する。学習時のモデルと一緒にすること(default: mlp)
* `-g <GPU index>` : 使用するGPUを指定する。-1の場合はCPUを使用(default: -1)
