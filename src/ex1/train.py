#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw
import six

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import serializers
from chainer.dataset import convert

# ヒント
# chainer links:
# L.Linear
#
# chainer functions:
# F.relu, F.tanh, F.sigmoid
# F.softmax_cross_entropy
#
# chainer optimizers:
# optimizers.Adam, optimizers.MomentumSGD
# optimizer methods:
# optimzier.setup(), optimizer,update()
#
# Variable methods:
# var.backward()

class ClassifierNet(chainer.Chain):
    def __init__(self):
        super(ClassifierNet, self).__init__(
            # TODO linkを追加する
            # 例: l1=L.Linear(2, 10),

        )

    def __call__(self, x, train=True):
        # TODO ニューラルネットワークの構成を定義する
        # 入力xのshapeは(batch_size, 2)
        # 出力のshapeは(batch_size, 2)


def main(train_x, train_y, test_x, test_y):
    # TODO 必要に応じてepoch_num, bach_sizeを修正
    epoch_num = 100
    batch_size = 10

    train_data = chainer.datasets.TupleDataset(train_x, train_y)
    test_data = chainer.datasets.TupleDataset(test_x, test_y)

    train_iterator = chainer.iterators.SerialIterator(train_data, batch_size)

    # TODO ニューラルネットワークの生成
    # net = ...

    # TODO optimizerの生成
    # optimizer = ...

    # TODO optimizerとモデルの関連付け


    train_loss_sum = 0
    train_acc_sum = 0
    train_num = 0
    while train_iterator.epoch < epoch_num:
        batch = train_iterator.next()
        x, t = convert.concat_examples(batch)
        # x: 入力データ shapeは(batch_size, 2)
        # t: 正解ラベル shapeは(batch_size,)

        # TODO ニューラルネットワークの実行
        # y = ...

        # TODO 損失の計算
        # loss = ...

        # TODO 精度の計算
        # acc = ...

        # TODO ネットワークの勾配初期化

        # TODO バックプロパゲーションを行い勾配を計算する

        # TODO パラメータを更新する


        train_loss_sum += float(loss.data) * len(x)
        train_acc_sum += float(acc.data) * len(x)
        train_num += len(x)
        if train_iterator.is_new_epoch:
            train_loss = train_loss_sum / train_num
            train_acc = train_acc_sum / train_num
            print('epoch {} done'.format(train_iterator.epoch))
            print('train loss: {} accracy: {}'.format(train_loss, train_acc))

            # TODO テストデータの精度を計算
            # コメント部分を有効にする
            # test_loss, test_acc = evaluate(net, test_data)
            # print('test loss: {} accuracy: {}'.format(test_loss, test_acc))

            train_loss_sum = 0
            train_acc_sum = 0
            train_num = 0

    save_image(net, train_x, train_y, test_x, test_y)

def evaluate(net, dataset):
    batch_size = 100
    iterator = chainer.iterators.SerialIterator(dataset, batch_size, repeat=False, shuffle=False)
    loss_sum = 0
    acc_sum = 0
    num = 0
    for batch in iterator:
        x, t = convert.concat_examples(batch)
        x = chainer.Variable(x, volatile=True)
        t = chainer.Variable(t, volatile=True)
        # TODO ネットワーク実行
        # y = ...

        # TODO 損失を計算
        # loss = ...

        # TODO 精度を計算
        # acc = ...


        n = len(x.data)
        loss_sum += float(loss.data) * n
        acc_sum += float(acc.data) * n
        num += n
    return loss_sum / num, acc_sum / num

def point_to_pixel(p):
    return (p[0] + 10) * 25, (10 - p[1]) * 25

def save_image(net, train_x, train_y, test_x, test_y):
    image = Image.new('RGB', (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    colors = [(255, 192, 192), (192, 192, 255)]
    for i in six.moves.range(100):
        x = np.zeros((100, 2), dtype=np.float32)
        x[:,1] = float(i) * 0.2 - 10 + 0.1
        x[:,0] = np.arange(100).astype(np.float32) * 0.2 - 10 + 0.1
        y = net(x, train=False)
        labels = y.data.argmax(axis=1)
        for p, label in zip(x, labels):
            pos_x, pos_y = point_to_pixel(p)
            draw.rectangle((pos_x - 2, pos_y - 2, pos_x + 2, pos_y + 2), colors[label])
    draw.line((250, 0, 250, 500), fill=0)
    draw.line((0, 250, 500, 250), fill=0)
    test_image = image.copy()

    colors = [(255, 0, 0), (0, 0, 255)]
    for x, y in zip(train_x, train_y):
        pos_x, pos_y = point_to_pixel(x)
        draw.ellipse((pos_x - 2, pos_y - 2, pos_x + 2, pos_y + 2), colors[y])
    image.save('image/result_train.png')

    draw = ImageDraw.Draw(test_image)
    for x, y in zip(test_x, test_y):
        pos_x, pos_y = point_to_pixel(x)
        draw.ellipse((pos_x - 2, pos_y - 2, pos_x + 2, pos_y + 2), colors[y])
    test_image.save('image/result_test.png')

def load_data(file_path):
    positions = []
    labels = []
    with open(file_path) as f:
        for line in f:
            x, y, label = line.split(',')
            positions.append([float(x), float(y)])
            labels.append(int(label))
    positions = np.asarray(positions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    return positions, labels

if __name__ == '__main__':
    train_x, train_y = load_data('data_train.txt')
    test_x, test_y = load_data('data_test.txt')
    main(train_x, train_y, test_x, test_y)
