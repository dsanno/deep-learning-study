#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import six
import time

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import optimizers
from chainer import serializers
from chainer.dataset import convert

import net as net_module

# Data augmentationのために
# データを-offset～offsetの範囲で平行移動する
def translate(x, offset):
    size = 28
    org_shape = x.shape
    batch_size = x.shape[0]
    xp = cuda.get_array_module(x)
    x = x.reshape((-1, 1, size, size))
    y = xp.zeros_like(x)
    y = y.reshape((-1, 1, size, size))
    offsets = xp.random.randint(-offset, offset + 1, (batch_size, 2))
    for i in six.moves.range(batch_size):
        left, top = offsets[i]
        right = min(size, size + left)
        bottom = min(size, size + top)
        left = max(0, left)
        top = max(0, top)
        y[i,:,size-bottom:size-top,size-right:size-left] = x[i,:,top:bottom,left:right]
    return y.reshape(org_shape)

def update(net, x, t, loss_func):
    y = net(x)
    loss = loss_func(y, t)

def evaluate(net, dataset, batch_size, device=None):
    # データを1回ずつ使用する場合はrepeat=Falseにする
    # そうしないと`for batch in iterator`が終了しない
    iterator = chainer.iterators.SerialIterator(dataset, batch_size, repeat=False, shuffle=False)
    loss_sum = 0
    acc_sum = 0
    num = 0
    for batch in iterator:
        raw_x, raw_t = convert.concat_examples(batch, device)
        # backpropagationは必要ないのでvolatileをTrueにする
        x = chainer.Variable(raw_x, volatile=True)
        t = chainer.Variable(raw_t, volatile=True)
        y = net(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        n = len(raw_x)
        loss_sum += float(loss.data) * n
        acc_sum += float(acc.data) * n
        num += n
    return loss_sum / num, acc_sum / num

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST training')
    parser.add_argument('--model', '-m', type=str, default='mlp', choices=['mlp', 'cnn'], help='Neural network model')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index, -1 indicates CPU')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Mini batch size')
    parser.add_argument('--prefix', '-p', type=str, default=None, help='prefix of saved file name')
    args = parser.parse_args()

    n_epoch = args.epoch
    batch_size = args.batch_size
    if args.prefix is None:
        prefix = args.model
    else:
        prefix = args.prefix
    if args.model == 'cnn':
        net = net_module.CNN()
    else:
        net = net_module.MLP(28 * 28, 10, 100)
    gpu_device = args.gpu
    if gpu_device >= 0:
        chainer.cuda.get_device(gpu_device).use()
        net.to_gpu(gpu_device)
        xp = cuda.cupy
    else:
        xp = np
    optimizer = optimizers.Adam()
    optimizer.setup(net)

    # MNISTデータセットを読み込む
    # get_mnistはMNISTデータセットファイルがなければダウンロードを行うので
    # 初回実行時は時間がかかる
    # データセットは"~/.chainer/dataset"以下に保存される
    train_data, test_data = chainer.datasets.get_mnist()
    # train dataとvalidation dataに分離する
    train_data, valid_data = chainer.datasets.split_dataset_random(train_data, len(train_data) - 5000)
    train_iterator = chainer.iterators.SerialIterator(train_data, batch_size)
    train_loss_sum = 0
    train_acc_sum = 0
    train_num = 0
    best_valid_acc = 0
    best_test_acc = 0
    last_clock = time.clock()

    while train_iterator.epoch < n_epoch:
        # 入力値と正解ラベルを取得
        # x: 入力値
        # t: 正解ラベル
        batch = train_iterator.next()
        x, t = convert.concat_examples(batch)
        x = xp.asarray(translate(x, 2))
        t = xp.asarray(t)
        # ネットワークの実行
        y = net(x)
        # 損失の計算
        loss = F.softmax_cross_entropy(y, t)
        # 精度の計算(学習時に必須ではない)
        acc = F.accuracy(y, t)
        # ネットワークの勾配初期化
        net.cleargrads()
        # バックプロパゲーションを行い勾配を計算する
        loss.backward()
        # パラメータを更新する
        optimizer.update()
        # 損失、精度の累積
        train_loss_sum += float(loss.data) * len(x)
        train_acc_sum += float(acc.data) * len(x)
        train_num += len(x)
        if train_iterator.is_new_epoch:
            train_loss = train_loss_sum / train_num
            train_acc = train_acc_sum / train_num
            valid_loss, valid_acc = evaluate(net, valid_data, batch_size, gpu_device)
            test_loss, test_acc = evaluate(net, test_data, batch_size, gpu_device)
            current_clock = time.clock()
            print('epoch {} done {}s elapsed'.format(train_iterator.epoch, current_clock - last_clock))
            last_clock = current_clock
            print('train loss: {} accuracy: {}'.format(train_loss, train_acc))
            print('valid loss: {} accuracy: {}'.format(valid_loss, valid_acc))
            print('test  loss: {} accuracy: {}'.format(test_loss, test_acc))
            train_acc_sum = 0
            train_num = 0
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                serializers.save_npz('{}.model'.format(prefix), net)
    train_iterator.finalize()

    print('best test accuracy: {}'.format(best_test_acc))
