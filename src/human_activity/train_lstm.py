# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import json
import numpy as np
import os
import six
import time

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import serializers
from chainer.dataset import convert

step_size = 10
window_size = 80
offset_size = 10

normal_actions = [
    'Bowing',
    'Clapping',
    'Handshaking',
    'Hugging',
    'Jumping',
    'Running',
    'Seating',
    'Standing',
    'Walking',
    'Waving',
]

def rotate(x, angle):
    xp = cuda.get_array_module(x)
    cos = xp.cos(-angle)
    sin = xp.sin(-angle)
    rot = xp.zeros((angle.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = -sin
    rot[:, 1, 0] = sin
    rot[:, 1, 1] = cos
    rot[:, 2, 2] = 1
    y = xp.zeros_like(x)
    for i in six.moves.range(len(y)):
        y[i, :, :, :] = xp.dot(rot[i], x[i]).transpose((1, 0, 2))
    return y

def preprocess(data):
    x, t = data
    n, ch, m = x.shape
    h = x.reshape((n, -1, 3, m))
    # subtract head x,y positions and constant z position
    base = h[:,0:1,:,0:1].copy()
    base[:, :, 2, :] = 0.8
    h = h - base
    # calculate body direction
    lankle_x = h[:, 6, 0, 0:1]
    lankle_y = h[:, 6, 1, 0:1]
    rankle_x = h[:, 8, 0, 0:1]
    rankle_y = h[:, 8, 1, 0:1]
    angle = np.arctan2(lankle_y - rankle_y, lankle_x - rankle_x)
    angle = angle.reshape((-1,))
    # normalize direction
    h = rotate(h, angle)
    return h.reshape(x.shape), t

def transform(x, scale_size, offset, rot_size):
    xp = cuda.get_array_module(x)
    n, ch, m = x.shape
    angle = (xp.random.random(n).astype(np.float32) - 0.5) * (np.pi / 180 * rot_size * 2)
    scale = 1 + (xp.random.random((n, 1, 1)).astype(np.float32) - 0.5) * scale_size * 2
    offsets = xp.random.uniform(-offset, offset, (n, 1, 1)).astype(np.float32)
    x = rotate(x.reshape((n, -1, 3, m)), angle).reshape((n, ch, m))
    return x * scale + offset

class ActionClassifier(chainer.Chain):

    def __init__(self):
        super(ActionClassifier, self).__init__(
            enc1=L.Linear(27, 50),
            enc2=L.Linear(50, 50),
            lstm1=L.LSTM(50, 50),
            dec1=L.Linear(50, 50),
            dec2=L.Linear(50, 10),
        )

    def __call__(self, x, train=True):
        h = F.relu(self.enc1(x))
        h = F.dropout(h, 0.5, train=train)
        h = F.relu(self.enc2(h))
        h = self.lstm1(h)
        h = F.relu(self.dec1(h))
        h = F.dropout(h, 0.5, train=train)
        h = self.dec2(h)
        return h

    def reset(self):
        self.lstm1.reset_state()

def read_file(file_path):
    data = np.loadtxt(file_path, usecols=six.moves.range(1, 28), dtype=np.float32)
    data = data[::step_size]
    row_num, col_num = data.shape
    n = (row_num - window_size) // offset_size + 1
    org_data = data
    data = np.zeros((n, window_size, col_num), dtype=np.float32)
    for i in six.moves.range(n):
        data[i, :, :] = org_data[i * offset_size:i * offset_size + window_size]
    data = data / 500
    return data.reshape((n, window_size, col_num)).transpose((0, 2, 1))

def read_data(data_dir, indices):
    xs = []
    ys = []
    for index in indices:
        sub_dir = 'sub{}'.format(index)
        for i, action_name in enumerate(normal_actions):
            file_name = '{}.txt'.format(action_name)
            path = os.path.join(data_dir, sub_dir, 'normal', file_name)
            x = read_file(path)
            xs.append(x)
            ys.append(np.full((x.shape[0],), i, dtype=np.int32))
        label_offset = len(normal_actions)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def evaluate(net, dataset, batch_size, device=None):
    iterator = chainer.iterators.SerialIterator(dataset, batch_size, repeat=False, shuffle=False)
    loss_sum = 0
    acc_sum = 0
    num = 0
    acc_table = np.zeros((10, 10), dtype=np.float32)
    for batch in iterator:
        raw_x, raw_t = convert.concat_examples(batch, device)
        t = chainer.Variable(raw_t, volatile=True)
        net.reset()
        for i in six.moves.range(raw_x.shape[2]):
            x = chainer.Variable(raw_x[:,:,i], volatile=True)
            y = net(x, train=False)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        n = len(raw_x)
        loss_sum += float(loss.data) * n
        acc_sum += float(acc.data) * n
        num += n
        selected = np.argmax(cuda.to_cpu(y.data), axis=1)
        for i, j in zip(raw_t, selected):
            acc_table[i, j] += 1
    return loss_sum / num, acc_sum / num, acc_table / np.sum(acc_table, axis=1, keepdims=True)

def train(epoch_num, batch_size, gpu_device, train_data, valid_data):
    net = ActionClassifier()
    if gpu_device >= 0:
        chainer.cuda.get_device(gpu_device).use()
        net.to_gpu(gpu_device)
        xp = cuda.cupy
    else:
        xp = np
    optimizer = optimizers.Adam()
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    train_dataset = chainer.datasets.TupleDataset(*train_data)
    valid_dataset = chainer.datasets.TupleDataset(*valid_data)
    train_iterator = chainer.iterators.SerialIterator(train_dataset, batch_size)

    train_loss_sum = 0
    train_acc_sum = 0
    train_num = 0
    last_clock = time.clock()
    log = {
        'log': [],
        'max_valid_acc': 0,
    }
    save_dir = 'result_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(save_dir)
    log_file_name = os.path.join(save_dir, 'log.txt')
    model_file_path = os.path.join(save_dir, 'action.model')

    while train_iterator.epoch < epoch_num:
        batch = train_iterator.next()
        x, t = convert.concat_examples(batch)
        x = transform(xp.asarray(x), 0.1, 0.2, 30)
        t = xp.asarray(t)
        net.reset()
        for i in six.moves.range(x.shape[2]):
            y = net(x[:,:,i])
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        net.cleargrads()
        loss.backward()
        optimizer.update()
        train_loss_sum += float(loss.data) * len(x)
        train_acc_sum += float(acc.data) * len(x)
        train_num += len(x)
        if train_iterator.is_new_epoch:
            train_loss = train_loss_sum / train_num
            train_acc = train_acc_sum / train_num
            valid_loss, valid_acc, acc_table = evaluate(net, valid_dataset, batch_size, gpu_device)
            current_clock = time.clock()
            print('epoch {} done {}s elapsed'.format(train_iterator.epoch, current_clock - last_clock))
            last_clock = current_clock
            print('train loss: {} accuracy: {}'.format(train_loss, train_acc))
            print('valid loss: {} accuracy: {}'.format(valid_loss, valid_acc))
            train_loss_sum = 0
            train_acc_sum = 0
            train_num = 0
            log['log'].append({
                'epoch': train_iterator.epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
            })
            if valid_acc > log['max_valid_acc']:
                log['max_valid_acc'] = valid_acc
                log['acc_table'] = acc_table.tolist()
                serializers.save_npz(model_file_path, net)
            with open(log_file_name, 'w') as f:
                json.dump(log, f, indent=4)
            if train_iterator.epoch == int(epoch_num * 0.5) or train_iterator.epoch == int(epoch_num * 0.75):
                optimizer.alpha *= 0.1 ** 0.5
    train_iterator.finalize()
    print('max valid accuracy: {}'.format(log['max_valid_acc']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physical action training')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index, -1 indicates CPU')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=50, help='Mini batch size')
    parser.add_argument('--data-dir', '-d', type=str, default='Vicon Physical Action Data Set', help='Dataset file directory')
    args = parser.parse_args()

    print('loading train data...')
    train_data = read_data(args.data_dir, [1, 2, 3, 4, 5, 6, 7])
    train_data = preprocess(train_data)
    print('train data num: {}'.format(len(train_data[0])))
    print('loading valid data...')
    valid_data = read_data(args.data_dir, [8, 9])
    valid_data = preprocess(valid_data)
    print('valid data num: {}'.format(len(valid_data[0])))
    print('start training')
    train(args.epoch, args.batch_size, args.gpu, train_data, valid_data)
