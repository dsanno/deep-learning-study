# -*- coding: utf-8 -*-

import argparse
import numpy as np
import six
import time

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import serializers

class Bouncing(chainer.Chain):

    def __init__(self):
        super(Bouncing, self).__init__(
            enc=L.Linear(2, 8),
            lstm=L.LSTM(8, 8),
            dec=L.Linear(8, 2),
        )

    def __call__(self, x, train=True):
        h1 = self.enc(x)
        h2 = self.lstm(h1)
        dx = self.dec(h2)
        return x + dx

    def reset(self):
        self.lstm.reset_state()

def make_train_data(size, length, fps=20):
    g = 9.8
    # initial velocity
    vx = np.random.random((size, 1, 1)).astype(np.float32) * 5 + 2.5
    vy = np.random.random((size, 1, 1)).astype(np.float32) * 5 + 2.5
    # elapsed time
    t = np.arange(length).astype(np.float32) / fps
    t = t.reshape((1, -1, 1)).repeat(size, axis=0)
    # calculate x, y positions
    x = vx * t
    interval = vy * 2 / g
    count = np.floor(t / interval)
    z = t - interval * count - interval * 0.5
    y = 0.5 * g * ((interval * 0.5) ** 2 - (z) ** 2)
    # output dimensions
    # (size, length, (x and y))
    return np.concatenate((x, y), axis=2)

def train_one(net, optimizer, xs):
    loss = 0
    h = None
    ts = xs[1:]
    update_interval = 10
    sum_loss = 0
    net.reset()
    for i in six.moves.range(3):
        x = xs[:, i, :]
        y = net(x, train=True)
    for i in six.moves.range(3, xs.shape[1] - 1):
        x = xs[:, i, :]
        t = xs[:, i + 1, :]
        y = net(x, train=True)
        loss += F.mean_squared_error(y, t)
        if (i + 1) % update_interval == 0:
            net.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data)
            loss.unchain_backward()
            loss = 0
    if float(loss.data) > 0:
        net.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data)
    return float(loss.data)

def train(epoch_num, model_path):
    batch_size = 5
    max_data_length = 100
    it_per_epoch = 100
    net = Bouncing()
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(net)
    last_clock = time.clock()
    for epoch in six.moves.range(epoch_num):
        loss = 0
        for it in six.moves.range(it_per_epoch):
            data_length = min((epoch // 10 + 1) * 10, max_data_length)
            x = make_train_data(batch_size, data_length)
            loss += train_one(net, optimizer, x)
        current_clock = time.clock()
        print('epoch {} done {}s elapsed'.format(epoch + 1, current_clock - last_clock))
        print('training loss: {}'.format(loss / it_per_epoch))
        last_clock = current_clock
        serializers.save_npz(model_path, net)
        if epoch + 1 == int(epoch_num * 0.5):
            optimizer.alpha *= 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bouncing ball training")
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model_path', '-m', type=str, default='bouncing.model', help='Model file path')
    args = parser.parse_args()
    train(args.epoch, args.model_path)
