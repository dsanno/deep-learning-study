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

latent_size = 30

class Generator(chainer.Chain):

    def __init__(self):
        initialW = chainer.initializers.Normal(0.02)
        super(Generator, self).__init__(
            fc1=L.Linear(latent_size, 3 * 3 * 128, initialW=initialW),
            deconv2=L.Deconvolution2D(128, 64, 3, stride=2, initialW=initialW),
            bn2=L.BatchNormalization(64),
            deconv3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1, initialW=initialW),
            bn3=L.BatchNormalization(32),
            deconv4=L.Deconvolution2D(32, 1, 4, stride=2, pad=1, initialW=initialW),
        )

    def __call__(self, x, train=True):
        h1 = F.reshape(F.relu(self.fc1(x)), (-1, 128, 3, 3))
        h2 = F.relu(self.bn2(self.deconv2(h1), test=not train))
        h3 = F.relu(self.bn3(self.deconv3(h2), test=not train))
        return self.deconv4(h3)


class Discriminator(chainer.Chain):

    def __init__(self):
        initialW = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__(
            conv1=L.Convolution2D(1, 32, 4, stride=2, pad=1, initialW=initialW),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(32, 64, 4, stride=2, pad=1, initialW=initialW),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 128, 3, stride=2, initialW=initialW),
            bn3=L.BatchNormalization(128),
            fc4=L.Linear(3 * 3 * 128, 1),
        )

    def __call__(self, x, train=True):
        h0 = F.reshape(x, (-1, 1, 28, 28))
        h1 = F.leaky_relu(self.bn1(self.conv1(h0), test=not train))
        h2 = F.leaky_relu(self.bn2(self.conv2(h1), test=not train))
        h3 = F.leaky_relu(self.bn3(self.conv3(h2), test=not train))
        return self.fc4(h3)


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

def update_generator(gen, dis, gen_optimizer, z):
    xp = cuda.get_array_module(z)

    # generate
    x_fake = gen(z)

    #discriminate
    y_fake = dis(x_fake)

    t_real = xp.ones_like(y_fake.data, dtype=np.int32)
    loss_gen = F.sigmoid_cross_entropy(y_fake, t_real)
    gen.cleargrads()
    loss_gen.backward()
    gen_optimizer.update()

    return loss_gen

def update_discriminator(gen, dis, dis_optimizer, x, z):
    xp = cuda.get_array_module(x)

    # generate
    x_fake = gen(z)

    #discriminate
    y_real = dis(x)
    y_fake = dis(x_fake)

    # target
    t_real = xp.ones_like(y_real.data, dtype=np.int32)
    t_fake = xp.zeros_like(y_fake.data, dtype=np.int32)
    loss_dis = F.sigmoid_cross_entropy(y_real, t_real) + F.sigmoid_cross_entropy(y_fake, t_fake)
    dis.cleargrads()
    loss_dis.backward()
    dis_optimizer.update()

    return loss_dis

def update(gen, dis, gen_optimizer, dis_optimizer, x, z):
    xp = cuda.get_array_module(x)

    # generate
    x_fake = gen(z)

    #discriminate
    y_real = dis(x)
    y_fake = dis(x_fake)

    # target
    t_real = xp.ones_like(y_real.data, dtype=np.int32)
    t_fake = xp.zeros_like(y_fake.data, dtype=np.int32)
    loss_dis = F.sigmoid_cross_entropy(y_real, t_real) + F.sigmoid_cross_entropy(y_fake, t_fake)
    dis.cleargrads()
    loss_dis.backward()
    dis_optimizer.update()

    t_real = xp.ones_like(y_fake.data, dtype=np.int32)
    loss_gen = F.sigmoid_cross_entropy(y_fake, t_real)
    gen.cleargrads()
    loss_gen.backward()
    gen_optimizer.update()

    return loss_gen, loss_dis

def generate_image(gen, z, file_path):
    x = gen(z, train=True)
    im = cuda.to_cpu(x.data)
    im = im.reshape((10, 10, 28, 28)).transpose((0, 2, 1, 3)).reshape((280, 280))
    im = ((im + 1) * 128).clip(0, 255).astype(np.uint8)
    Image.fromarray(im).save(file_path)

def train(gen, dis, gen_optimizer, dis_optimizer, iterator, iteration, name):
    xp = gen.xp
    loss_gen_sum = 0
    loss_dis_sum = 0
    loss_num = 0
    test_z1 = xp.random.uniform(-1, 1, (10, 1, latent_size)).astype(np.float32)
    test_z2 = xp.random.uniform(-1, 1, (10, 1, latent_size)).astype(np.float32)
    weights = xp.arange(10).astype(np.float32).reshape((1, 10, 1)).repeat(10, axis=0) / 9
    test_z = (1 - weights) * test_z1 + weights * test_z2
    test_z = test_z.reshape((-1, latent_size))
    last_clock = time.clock()
    for i in six.moves.range(iteration):
        batch = train_iterator.next()
        x = convert.concat_examples(batch)
        x = xp.asarray(x - 1)
        z = xp.random.uniform(-1, 1, (x.shape[0], latent_size)).astype(np.float32)
        loss_gen, loss_dis = update(gen, dis, gen_optimizer, dis_optimizer, x, z)
        loss_dis_sum += float(loss_dis.data)
        loss_gen_sum += float(loss_gen.data)
        loss_num += 1

        if (i + 1) % 100 == 0:
            current_clock = time.clock()
            print('iteration {} done {}s elapsed'.format(i + 1, current_clock - last_clock))
            last_clock = current_clock
            print('gen loss: {}'.format(loss_gen_sum / loss_num))
            print('dis loss: {}'.format(loss_dis_sum / loss_num))
            loss_gen_sum = 0
            loss_dis_sum = 0
            loss_num = 0
            generate_image(gen, test_z, '{0}_{1:06d}.png'.format(name, i + 1))
            serializers.save_npz('{}_gen.model'.format(name), gen)
            serializers.save_npz('{}_dis.model'.format(name), dis)
    train_iterator.finalize()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generative Adversarial Net sample')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index, -1 indicates CPU')
    parser.add_argument('--iter', '-i', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Mini batch size')
    parser.add_argument('--name', '-n', type=str, default='gan', help='saved file name')
    args = parser.parse_args()

    batch_size = args.batch_size
    gen = Generator()
    dis = Discriminator()
    gpu_device = args.gpu
    if gpu_device >= 0:
        chainer.cuda.get_device(gpu_device).use()
        gen.to_gpu(gpu_device)
        dis.to_gpu(gpu_device)
        xp = cuda.cupy
    else:
        xp = np
    gen_optimizer = optimizers.Adam(0.0002, beta1=0.5)
    gen_optimizer.setup(gen)
    dis_optimizer = optimizers.Adam(0.0002, beta1=0.5)
    dis_optimizer.setup(dis)

    train_data, test_data = chainer.datasets.get_mnist(False, scale=2)
    train_iterator = chainer.iterators.SerialIterator(train_data, batch_size)
    train(gen, dis, gen_optimizer, dis_optimizer, train_iterator, args.iter, args.name)
