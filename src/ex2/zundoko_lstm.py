# -*- coding: utf-8 -*-

import numpy as np
import six

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import Variable

np.random.seed()
zun = 0
doko = 1
input_num = 2
none = 0
kiyoshi = 1
output_num = 2
input_words = ['ズン', 'ドコ']
output_words = [None, '＼キ・ヨ・シ！／']
# for Windows command prompt
#input_words = ['Zun', 'Doko']
#output_words = [None, 'Ki.Yo.Shi!']
hidden_num = 8
update_iteration = 20

class Zundoko(chainer.Chain):
    def __init__(self):
        super(Zundoko, self).__init__(
            word=L.EmbedID(input_num, hidden_num),
            lstm = L.LSTM(hidden_num, hidden_num),
            out=L.Linear(hidden_num, output_num),
        )

    def __call__(self, x, prev_h=None, train=True):
        h0 = self.word(x)
        # y = ...
        return y

kiyoshi_list = [zun, zun, zun, zun, doko]
kiyoshi_pattern = 0
kiyoshi_mask = (1 << len(kiyoshi_list)) - 1
for token in kiyoshi_list:
    kiyoshi_pattern = (kiyoshi_pattern << 1) | token

zundoko = Zundoko()
optimizer = optimizers.Adam(alpha=0.01)
optimizer.setup(zundoko)

def train():
    loss = 0
    acc = 0
    batch_size = 20
    recent_pattern = np.zeros((batch_size,), dtype=np.int32)
    h = None
    for i in six.moves.range(200):
        x = np.random.randint(0, input_num, batch_size).astype(np.int32)
        y = zundoko(x, train=True)
        recent_pattern = ((recent_pattern << 1) | x) & kiyoshi_mask
        if i < len(kiyoshi_list):
            t = np.full((batch_size,), none, dtype=np.int32)
        else:
            t = np.where(recent_pattern == kiyoshi_pattern, kiyoshi, none).astype(np.int32)
        loss += F.softmax_cross_entropy(y, t)
        acc += float(F.accuracy(y, t).data)
        if (i + 1) % update_iteration == 0:
            zundoko.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            print('train loss: {} accuracy: {}'.format(loss.data, acc / update_iteration))
            loss = 0
            acc = 0

def predict():
    batch_size = 1
    h = None
    for i in six.moves.range(200):
        x = np.random.randint(0, input_num, batch_size).astype(np.int32)
        x = Variable(x, volatile=True)
        y = zundoko(x, train=False)
        print input_words[x.data[0]]
        out_word = output_words[np.argmax(y.data[0])]
        if out_word != None:
            print(out_word)
            break

for iteration in range(25):
    train()

for i in range(5):
    print('prediction: {}'.format(i + 1))
    predict()
