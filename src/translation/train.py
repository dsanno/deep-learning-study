import argparse
import numpy as np
import os
import six
import time

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import serializers

pickle = six.moves.cPickle

begin_id = 0
end_id = 1

class EncoderDecoder(chainer.Chain):

    def __init__(self, input_size, output_size, hidden_size=256):
        super(EncoderDecoder, self).__init__(
            enc_embed=L.EmbedID(input_size, hidden_size, ignore_label=-1),
            dec_embed=L.EmbedID(output_size, hidden_size, ignore_label=-1),
            enc1=L.LSTM(hidden_size, hidden_size),
            enc2=L.LSTM(hidden_size, hidden_size),
            dec1=L.LSTM(hidden_size, hidden_size),
            dec2=L.LSTM(hidden_size, hidden_size),
            dec_out=L.Linear(hidden_size, output_size),
        )
        self.c = None
        self.h = None

    def __call__(self, x, decode=False, train=True):
        if decode:
            return self.decode(x, train=train)
        return self.encode(x, train=train)

    def encode(self, x, train=True):
        h = self.enc_embed(x)
        h = self.enc1(h)
        h = self.enc2(h)
        return h

    def decode(self, x, train=True):
        if self.dec1.c is None:
            self.dec1.set_state(self.enc1.c, self.enc1.h)
            self.dec2.set_state(self.enc2.c, self.enc2.h)
        h = self.dec_embed(x)
        h = self.dec1(h)
        h = self.dec2(h)
        h = F.dropout(h, train=train)
        return self.dec_out(h)

    def reset_state(self):
        self.enc1.reset_state()
        self.enc2.reset_state()
        self.dec1.reset_state()
        self.dec2.reset_state()

    def permutate(self, order):
        for link in [self.enc1, self.enc2]:
            link.c = F.permutate(link.c, order)
            link.h = F.permutate(link.h, order)

    def get_sub_state(self, index):
        state = []
        for link in [self.dec1, self.dec2]:
            c = None
            h = None
            if link.c is not None:
                c = link.c.data[index, ...].copy()
            if link.h is not None:
                h = link.h.data[index, ...].copy()
            state.append((c, h))
        return tuple(state)

    def set_sub_state(self, index, state):
        for s, link in zip(state, [self.dec1, self.dec2]):
            c, h = s
            if c is not None and link.c is not None:
                link.c.data[index, ...] = c
            if h is not None and link.h is not None:
                link.h.data[index, ...] = h

def sort_batch(inputs, targets):
    input_order = np.argsort(np.asarray(list(map(len, inputs)), dtype=np.int32))[::-1]
    target_order = np.argsort(np.asarray(list(map(len, targets)), dtype=np.int32))[::-1]
    sorted_inputs = [inputs[i] for i in input_order]
    sorted_targets = [targets[i] for i in target_order]
    inv_input_order = np.zeros_like(input_order, dtype=np.int32)
    inv_input_order[input_order] = np.arange(len(input_order), dtype=np.int32)
    return sorted_inputs, sorted_targets, inv_input_order[target_order]

def concat_examples(examples):
    inputs, targets = zip(*examples)
    return sort_batch(inputs, targets)

def forward(net, xs, ts, permutation, train=True):
    xp = net.xp
    loss = 0
    acc = 0
    num = 0
    net.reset_state()
    for x in F.transpose_sequence(xs):
        v = chainer.Variable(xp.asarray(x.data), volatile=not train)
        net(v, decode=False, train=train)
    net.permutate(permutation)
    t_seq = F.transpose_sequence(ts)
    t = xp.asarray(t_seq[0].data)
    for next_t in t_seq[1:]:
        n = len(next_t.data)
        y = net(t[:n], decode=True, train=train)
        next_t = xp.asarray(next_t.data)
        loss += F.softmax_cross_entropy(y, next_t) * n
        acc += F.accuracy(y, next_t) * n
        num += n
        t = next_t
    return num, loss, acc

def evaluate(net, inputs, targets, batch_size):
    dataset = chainer.datasets.TupleDataset(inputs, targets)
    iterator = chainer.iterators.SerialIterator(dataset, batch_size, shuffle=False, repeat=False)
    loss_sum = 0
    acc_sum = 0
    total_num = 0
    for batch in iterator:
        xs, ts, permutation = concat_examples(batch)
        num, loss, acc = forward(net, xs, ts, permutation, train=False)
        loss_sum += float(loss.data)
        acc_sum += float(acc.data)
        total_num += float(num)
    return total_num, loss_sum, acc_sum

def test(net, inputs, test_token_len, beam_width=10):
    xp = net.xp
    from_sentences = []
    sentences = []
    for xs in inputs:
        net.reset_state()
        for raw_x in xs.data:
            x = xp.full((beam_width,), raw_x, dtype=np.int32)
            x = chainer.Variable(x, volatile=True)
            net(x, decode=False, train=False)
        candidates = [(None, [begin_id], 0)]
        for i in six.moves.range(test_token_len):
            next_candidates = []
            current_candidates = []
            x = []
            for sub_state, tokens, likelihood in candidates:
                if tokens[-1] == end_id:
                    continue
                if sub_state != None:
                    net.set_sub_state(len(x), sub_state)
                current_candidates.append((len(x), tokens, likelihood))
                x.append(tokens[-1])
            x = chainer.Variable(xp.asarray(x, dtype=np.int32), volatile=True)
            y = F.log_softmax(net(x, decode=True, train=False))
            for j, tokens, likelihood in current_candidates:
                sub_state = net.get_sub_state(j)
                token_likelihoods = cuda.to_cpu(y.data[0])
                top_tokens = token_likelihoods.argsort()[-beam_width:]
                next_candidates.extend([(sub_state, tokens + [j], likelihood + token_likelihoods[j]) for j in top_tokens])
            candidates = sorted(next_candidates, key=lambda x: -x[2])[:beam_width]
            if all([candidate[1][-1] == end_id for candidate in candidates]):
                break
        sentences.append(candidates[0][1][1:-1])
    return sentences

    for xs in inputs:
        while len(tokens) < test_token_len:
            token_id = chainer.Variable(xp.asarray([token_id], dtype=np.int32), volatile=True)
            y = net(token_id, decode=True, train=False)
            token_id = int(xp.argmax(y.data[0]))
            if token_id == end_id:
                break
            tokens.append(token_id)
        sentences.append(tokens)
    return sentences

def output_result(sentences, words, output_path, epoch):
    base, ext = os.path.splitext(output_path)
    with open('{0}_{1:04d}{2}'.format(base, epoch, ext), 'w') as f:
        for token_ids in sentences:
            tokens = map(lambda x: words[x], token_ids)
            f.write(' '.join(tokens) + '\n')

def train(net, optimizer, inputs, targets, epoch_num, batch_size, output_path, test_words=None, test_inputs=None, test_targets=None, test_output_path=None, test_token_len=20):
    xp = net.xp
    train_dataset = chainer.datasets.TupleDataset(inputs, targets)
    train_iterator = chainer.iterators.SerialIterator(train_dataset, batch_size)
    train_loss_sum = 0
    train_acc_sum = 0
    train_num = 0
    last_clock = time.clock()
    while train_iterator.epoch < epoch_num:
        batch = train_iterator.next()
        xs, ts, permutation = concat_examples(batch)
        num, loss, acc = forward(net, xs, ts, permutation)
        net.cleargrads()
        loss /= num
        loss.backward()
        optimizer.update()
        train_loss_sum += float(loss.data) * num
        train_acc_sum += float(acc.data)
        train_num += float(num)
        if train_iterator.is_new_epoch:
            print('epoch {} done'.format(train_iterator.epoch))
            print('train loss: {}'.format(train_loss_sum / train_num))
            if test_inputs is not None and test_targets is not None:
                test_num, test_loss, test_acc = evaluate(net, test_inputs, test_targets, batch_size)
            print('train acc:  {}'.format(train_acc_sum / train_num))
            print('test loss: {}'.format(test_loss / test_num))
            print('test acc:  {}'.format(test_acc / test_num))
            if test_words is not None and test_inputs is not None and test_output_path is not None:
                test_sentences = test(net, test_inputs, test_token_len)
                output_result(test_sentences, test_words, test_output_path, train_iterator.epoch)
            current_clock = time.clock()
            print('{}s elapsed'.format(current_clock - last_clock))
            last_clock = current_clock
            train_loss_sum = 0
            train_acc_sum = 0
            train_num = 0
            serializers.save_npz(output_path, net)
            if train_iterator.epoch >= 12 and (train_iterator.epoch - 12) % 8 == 0:
                optimizer.alpha *= 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation training')
    parser.add_argument('data_path', type=str, help='Dataset file path')
    parser.add_argument('input_lang', type=str, help='Input language')
    parser.add_argument('output_lang', type=str, help='Output language')
    parser.add_argument('output_model', type=str, help='Output model file path')
    parser.add_argument('test_result', type=str, help='Test result file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index, -1 indicates CPU')
    parser.add_argument('--epoch', '-e', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Mini batch size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Model hidden layer size')
    parser.add_argument('--max_result_len', type=int, default=20, help='Maximum test result token length')
    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)
    input_words = dataset[args.input_lang]['words']
    input_train = dataset[args.input_lang]['train']
    input_test = dataset[args.input_lang]['test']
    output_words = dataset[args.output_lang]['words']
    output_train = dataset[args.output_lang]['train']
    output_test = dataset[args.output_lang]['test']

    inputs = list(map(lambda x: chainer.Variable(np.asarray(x[::-1], dtype=np.int32), volatile=True), input_train))
    max_output_len = max(map(len, output_train))
    targets = list(map(lambda x: chainer.Variable(np.asarray([begin_id] + x + [end_id], dtype=np.int32), volatile=True), output_train))
    test_inputs = list(map(lambda x: chainer.Variable(np.asarray(x[::-1], dtype=np.int32), volatile=True), input_test))
    test_targets = list(map(lambda x: chainer.Variable(np.asarray([begin_id] + x + [end_id], dtype=np.int32), volatile=True), output_test))

    net = EncoderDecoder(len(input_words), len(output_words), args.hidden_size)
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(net)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        net.to_gpu(args.gpu)
    train(net, optimizer, inputs, targets, args.epoch, args.batch_size, args.output_model, test_words=output_words, test_inputs=test_inputs, test_targets=test_targets, test_output_path=args.test_result, test_token_len=args.max_result_len)
