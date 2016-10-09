# Caffe modelの使用

## Caffe modelとは
Deep Learningフレームワークである[Caffe](http://caffe.berkeleyvision.org/)を使って学習したモデルである。多数のモデルファイルが公開されており、研究用途で使われることが多い。画像分類に使われるモデルをそのまま使用したり、画像の特徴量を抽出して別の用途(例えば物体検出、キャプション生成など)に使用することができる。

## Caffe modelの属性の調べ方

Caffe modelはprototextファイルに記述されている。
prototxt内のレイヤー定義については[Caffe公式ドキュメントのLayers](http://caffe.berkeleyvision.org/tutorial/layers.html)を参照すること。

例えば[GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)の[train_val.prototxt](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt)から次のことがわかる

* 入力は`data`レイヤー
* `transform_param`から入力の画像サイズは224、BGRの平均値がそれぞれ104, 117, 123である
* 出力は`loss3/classifier`

## Caffe modelの読み込み

caffemodelファイルへのパスを指定して`chainer.functions.caffe.CaffeFunction`のインスタンスを生成する。

例:

```
from chainer.functions import caffe

model = caffe.CaffeFunction('bvlc_googlenet.caffemodel')
```

## Caffe modelの実行

生成したインスタンスを関数として呼び出す
* 入力
    * `inputs`: 入力データを`dict`形式で渡す。`dict`のkeyはモデルに依存する
    * `outputs`: 出力レイヤー名の`list`
    * `disable`: 使用しないレイヤー名の`list`を渡す。出力レイヤーに関与しないレイヤーがある場合に指定する
    * `train`: Trueなら学習モード、それ以外はテストモードになる。back propagationが必要な場合はTrueを指定する
* 出力
    * 指定したレイヤーの出力の`tuple`

例:

```
y, = model(
    inputs={'data': chainer.Variable(x, volatile=True)},
    outputs=['loss3/classifier'],
    disable=['loss1/ave_pool', 'loss2/ave_pool'],
    train=False)
```

* 入力はnumpyもしくはcupy配列である
* 入力のshapeは(mini_batch, color, height, width)であることが多い。
* 入力からは平均画像を引く必要がある。
平均画像はモデルによって異なる。
* 色空間はRGBではなくBGRであることが多い。

## サンプルプログラム

### ソースディレクトリ

(root dir)/src/caffemodel

### 実行方法

```
$ cd src/caffemodel
$ python predict.py image_dir
```

オプション:
* `-m <caffemodel path>` : GoogleNetモデルのファイルパス(default: bvlc_googlenet.caffemodel)
* `-l <label file path>` : ラベルファイルのパス。ソースツリーに含まれるのでデフォルト値で問題ない(default: labels.txt)
* `-g <GPU index>` : 使用するGPUを指定する。-1の場合はCPUを使用(default: -1)

## よく使用されるCaffe model

* [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)  
様々なCaffe modelへのリンクがある。
* [Residual Network](https://github.com/KaimingHe/deep-residual-networks)  
ILSVRC 2015 で使用されたResidual Network。
* [VGG 16-layers](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)  
ILSVRC 2014 で使用されたVGG 16-layersモデル。
* [VGG 19-layers](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)
ILSVRC 2014 で使用されたVGG 19-layersモデル。
* [GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
ILSVRC 2014 で使用されたGoogleNetモデル。
* [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)  
LSVRC-2010 で使用されたAlexNetモデル。
