#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import functions as F
from chainer import links as L

# Fully Connected LayerのみのMulti Layer Perceptron
class MLP(chainer.Chain):

    def __init__(self, n_in, n_out, n_hidden):
        # Networkの持つLinkを定義する
        # superクラスの__ini__を使う方法と
        # add_link()メソッドで追加する方法とがある
        # Chainer1.12からL.Linear()の第1引数はNoneでよく、
        # その場合入力サイズは実際のデータから自動的に決まる
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_hidden),
            l2=L.Linear(n_hidden, n_hidden),
            l3=L.Linear(n_hidden, n_out),
        )

    def __call__(self, x, train=True):
        # Linkはcallableとなっており、関数として呼び出すとLinkの処理
        # L.LinearはWx + bを計算する(W, bはL.Linearが持つ重み行列とバイアス項)
        h = self.l1(x)
        # Dropoutを実行する
        # Dropoutは学習時と予測時とで挙動が異なるので、trainでどちらであるかを指定する必要がある
        h = F.dropout(h, 0.5, train=train)
        # ReLUを実行する
        h = F.relu(h)
        h = self.l2(h)
        h = F.dropout(h, 0.5, train=train)
        h = F.relu(h)
        h = self.l3(h)
        return h

# Convolutional Neural Network
class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 16, 5),
            conv2=L.Convolution2D(16, 32, 3, pad=1),
            conv3=L.Convolution2D(32, 64, 3, pad=1),
            bn1=L.BatchNormalization(16),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(64),
            fc4=L.Linear(3 * 3 * 64, 128),
            fc5=L.Linear(128, 10),
        )

    def __call__(self, x, train=True):
        # L.Convolution2Dの入力は4次元である必要があるのでreshapeする
        # 各軸は(batch, channel, y, x)を意味する
        h = F.reshape(x, (-1, 1, 28, 28))
        h = self.conv1(h)
        h = self.bn1(h, test=not train)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = self.bn2(h, test=not train)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv3(h)
        h = self.bn3(h, test=not train)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.fc4(h)
        h = F.dropout(h, train=train)
        h = F.relu(h)
        h = self.fc5(h)
        return h
