#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import six
from PIL import Image

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer.serializers import npz

import net as net_module

def predict(net, image):
    # 予測時にはtrain=Falseを指定する
    y = net(chainer.Variable(image, volatile=True), train=False)
    return F.softmax(y).data

def main(args):
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
    npz.load_npz(args.model_file, net)
    image = Image.open(args.image_file).convert('L').resize((28, 28), Image.BILINEAR)
    # 学習データは値の範囲が0～1なのでそれに合わせるために255で割る
    # 学習データは背景が0なので反転する
    image = 1 - xp.asarray(image).astype(np.float32) / 255
    image = image.reshape((1, -1))
    probs = cuda.to_cpu(predict(net, image))[0]
    results = sorted(zip(six.moves.range(10), probs), key=lambda x: -x[1])
    for n, p in results:
        print('{0:d}: {1:.4f}'.format(n, p))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST prediction')
    parser.add_argument('model_file', type=str, help='Model file path')
    parser.add_argument('image_file', type=str, help='Image file path')
    parser.add_argument('--model', '-m', type=str, default='mlp', choices=['mlp', 'cnn'], help='Neural network model')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index, -1 indicates CPU')
    args = parser.parse_args()

    main(args)
