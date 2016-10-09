# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from PIL import Image

import chainer
from chainer import cuda
from chainer import functions as F
from chainer.functions import caffe


def load_image(file_path, device=-1):
    mean = np.asarray([104, 117, 123], dtype=np.float32)
    # 中央の正方形領域を切り抜き、入力サイズを224 x 224
    # 入力サイズはモデルによって異なる
    image = Image.open(file_path).convert('RGB')
    w, h = image.size
    if w > h:
        offset_x = (w - h) // 2
        offset_y = 0
        size = h
    else:
        offset_x = 0
        offset_y = (h - w) // 2
        size = w
    image = image.crop((offset_x, offset_y, size, size)).resize((224, 224))
    # RGBからBGRに変換する
    # caffemodelは通常BGRを入力とする
    x = np.asarray(image, dtype=np.float32)[:,:,::-1] - mean
    # shapeを(batch, channel, height, width)に変換
    x = x.transpose((2, 0, 1)).reshape((1, 3, 224, 224))
    # 必要ならcupy配列に変換
    if device >= 0:
        x = cuda.to_gpu(x, device)
    return x

def predict(model, x):
    # caffemodelインスタンスを関数として呼ぶことでレイヤーの出力を得ることができる。
    # 入力はinputsでdict形式で指定する
    # 出力はレイヤー名のリストを指定する
    # disableパラメータで使用しないレイヤーを指定できる。出力に関与しないレイヤーを使用しないことで無駄な計算を抑えることができる。
    y, = model(inputs={'data': chainer.Variable(x, volatile=True)}, outputs=['loss3/classifier'], disable=['loss1/ave_pool', 'loss2/ave_pool'], train=False)
    return F.softmax(y)

def is_image_file(path):
    base, ext = os.path.splitext()
    os.path.isfile(path)

def print_top(x, categories, top=10):
    scores = x.reshape((-1,))
    result = sorted(zip(scores, categories), reverse=True)
    for i, (score, label) in enumerate(result[:top]):
        print '{:>3d} {:>6.2f}% {}'.format(i + 1, float(score) * 100, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict image category')
    parser.add_argument('image_path', type=str, help='Image file  or directory path')
    parser.add_argument('--model', '-m', type=str, default='bvlc_googlenet.caffemodel', help='caffe model file path')
    parser.add_argument('--label', '-l', type=str, default='labels.txt', help='label file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index. negative value indicates CPU')
    args = parser.parse_args()

    device = args.gpu
    categories = np.loadtxt(args.label, str, delimiter='\n')
    caffe_model = caffe.CaffeFunction(args.model)
    if device >= 0:
        caffe_model.to_gpu(device)

    if os.path.isdir(args.image_path):
        image_files = os.listdir(args.image_path)
        image_paths = map(lambda f: os.path.join(args.image_path, f), image_files)
        image_paths = filter(os.path.isfile, image_paths)
    else:
        image_paths = [args.image_path]

    for image_path in image_paths:
        try:
            x = load_image(image_path, device)
            y = predict(caffe_model, x)
            print(os.path.basename(image_path))
            # cupy配列はソートに時間がかかるのでnumpy配列に変換する
            print_top(cuda.to_cpu(y.data), categories)
        except IOError:
            print('cannot load {}'.format(image_file))
