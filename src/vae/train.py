#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np
import six
import time
from PIL import Image

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import serializers
from chainer.dataset import convert

from net import VAE

latent_size = 30


# for data augmentation
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

def update(net, optimizer, x):
    xp = cuda.get_array_module(x)
    div_weight = 1

    y, mean, var = net(x)
    loss = F.mean_squared_error(x, y) + div_weight * F.gaussian_kl_divergence(mean, var) / float(y.size)
    net.cleargrads()
    loss.backward()
    optimizer.update()
    return loss

def generate_image(net, z, file_path):
    x = net.generate(z)
    im = cuda.to_cpu(x.data)
    im = im.reshape((10, 10, 28, 28)).transpose((0, 2, 1, 3)).reshape((280, 280))
    im = ((1 - im) * 256).clip(0, 255).astype(np.uint8)
    Image.fromarray(im).save(file_path)

def train(net, optimizer, iterator, iteration, name):
    xp = net.xp
    loss_sum = 0
    loss_num = 0
    test_z1 = xp.random.uniform(-1, 1, (10, 1, latent_size)).astype(np.float32)
    test_z2 = xp.random.uniform(-1, 1, (10, 1, latent_size)).astype(np.float32)
    weights = xp.arange(10).astype(np.float32).reshape((1, 10, 1)).repeat(10, axis=0) / 9
    test_z = (1 - weights) * test_z1 + weights * test_z2
    test_z = test_z.reshape((-1, latent_size))
    last_clock = time.clock()
    for i in six.moves.range(iteration):
        batch = train_iterator.next()
        x, t = convert.concat_examples(batch)
        x = xp.asarray(translate(x, 2))
        loss = update(net, optimizer, x)
        loss_sum += float(loss.data)
        loss_num += 1

        if (i + 1) % 1000 == 0:
            current_clock = time.clock()
            print('iteration {} done {}s elapsed'.format(i + 1, current_clock - last_clock))
            last_clock = current_clock
            print('loss: {}'.format(loss_sum / loss_num))
            loss_sum = 0
            loss_num = 0
            generate_image(net, test_z, '{0}_{1:06d}.png'.format(name, i + 1))
            serializers.save_npz('{}.model'.format(name), net)
    train_iterator.finalize()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Auto Encoder sample')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index, -1 indicates CPU')
    parser.add_argument('--iter', '-i', type=int, default=30000, help='Number of iterations')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Mini batch size')
    parser.add_argument('--name', '-n', type=str, default='image/vae', help='saved file name')
    args = parser.parse_args()

    batch_size = args.batch_size
    net = VAE()
    gpu_device = args.gpu
    if gpu_device >= 0:
        chainer.cuda.get_device(gpu_device).use()
        net.to_gpu(gpu_device)
        xp = cuda.cupy
    else:
        xp = np
    optimizer = optimizers.Adam()
    optimizer.setup(net)

    train_data, test_data = chainer.datasets.get_mnist()
    train_iterator = chainer.iterators.SerialIterator(train_data, batch_size)
    train(net, optimizer, train_iterator, args.iter, args.name)
