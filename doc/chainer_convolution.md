# Convolutional Neural Network (CNN, 畳み込みネットワーク)

Convolutional Neural Networkは、Convolutional Layerを持ったNeural Networkを指す。
Convolutional LayerとReLU等のActivation Functionを組み合わせてNeural Networkを構築することが多い。
以降は2次元のConvolutional Layerについて説明する。

Convolutional Layerの図つきの説明としては以下がわかりやすい。
* [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)
* [Convolutional Neural Networks (CNNs): An Illustrated Explanation](http://xrds.acm.org/blog/2016/06/convolutional-neural-networks-cnns-illustrated-explanation/)

## Convolutional Layer (畳み込み層)

### 処理の概要

* 入力データは`(入力チャンネル数, 高さ, 幅)`の3次元。(ミニバッチも考慮すると`(ミニバッチ数, チャンネル数, 高さ, 幅)`の4次元となる)
* 入力データから`(入力チャンネル数, kh, kw)`の領域を取り出す。ここで`kh`と`kw`はカーネルと呼ばれる領域の高さと幅である。
* 取り出したデータをベクトル化し(1列に並べ)、行列`W`を掛けてバイアス`b`を足す。行列のサイズは`(出力チャンネル数, 入力チャンネル数 * kh * kw)`となる。
* 上記の操作を、取り出す位置を一定間隔ずつずらしながら行う。このとき使用する`W`と`b`は共通である。
* 出力したベクトルを並べる。出力は`(出力チャンネル数, 出力高さ, 出力幅)`となる。出力の高さと幅については後述する。

### Convolutional Layerの特徴

* 局所的な特徴を抽出することができる。
* `W`と`b`が共通なので、Fully Connected Layerに比べるとLayerのパラメータサイズが小さくなる。

### Convolutional Layerの属性

Convolutional Layerは以下の属性を持つ。

* 入力チャンネル数、出力チャンネル数  
入力と出力の特徴量の大きさを表す。例えばRGB値を持つカラー画像であればチャンネル数は3となる。
* カーネルサイズ  
畳み込み演算を実行する領域(カーネル)の幅と高さを表す。
* stride  
カーネルを移動する間隔である。
* パディング
入力データの高さと幅方向の上下に挿入するパディングである。通常はパディングした位置には0を挿入する。

出力データの幅と高さは以下のようになる。

```
# in_width, in_height: 入力高さ
# in_height, out_height: 入力高さ
# pad_w, pad_h: パディング幅、高さ
# kernel_w, kernel_h: カーネル幅、高さ
# stride_w, stride_h: stride幅、高さ

out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1
out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1
```

Convolutional LayerでForward propagationを行うコードは以下のようになる。(カーネルサイズ、stride、パディングは高さ方向と幅方向で同じにしてある。)

```
import numpy as np
import six

def convolution_2d(x, w, ksize, stride=1, pad=0):
    batch_size, in_channel, width, height = x.shape
    out_channel = w.shape[0]

    if pad > 0:
        h = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    else:
        h = x

    out_height = (height + 2 * pad - ksize) // stride + 1
    out_width = (width + 2 * pad - ksize) // stride + 1
    y = np.zeros((batch_size, out_channel, out_height, out_width), dtype=x.dtype)

    for i in six.moves.range(batch_size):
        for j in six.moves.range(out_height):
            for k in six.moves.range(out_width):
                feature = np.ravel(h[i, :, j * stride:j * stride + ksize, k * stride:k * stride + ksize])
                y[i, :, j, k] = np.dot(w, feature)

    return y
```

## ChainerでのConvolution Layerの利用

`chainer.links.Convolution2D`または`chainer.links.ConvolutionND`を使う。
`Convolution2D`の入力配列のサイズは`(ミニバッチ数, チャンネル数, 高さ, 幅)`の4次元にする必要がある。`ConvolutionND`の入力配列のサイズは`(ミニバッチ数, チャンネル数, 各次元のサイズ)`のN+2次元にする必要がある。
