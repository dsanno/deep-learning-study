import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import six

from chainer import serializers

def update(index, scat, items):
    scat.set_offsets(items[:, index, :])
    return scat

def plot_animation(data, interval=50):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Bouncing Ball')

    scat = ax.scatter(data[:, 0, 0], data[:, 0, 1], c=['b', 'r'], s=50, alpha=1)
    # Creating the Animation object
    max_frame = data.shape[1]
    anim = animation.FuncAnimation(
        fig, update, max_frame, fargs=(scat, data),
        interval=interval, blit=False
    )
    plt.show()

def make_data(size, length, vx=5.0, vy=5.0, fps=20):
    g = 9.8
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

def predict(net, xs):
    result = np.zeros_like(xs)
    result[:, 0, :] = xs[:, 0, :]
    net.reset()
    for i in six.moves.range(3):
        x = xs[:, i, :]
        y = net(x, train=False)
        result[:, i + 1, :] = y.data[:, :]
    y = xs[:, 3, :]
    for i in six.moves.range(3, xs.shape[1] - 1):
        y = net(y, train=False)
        result[:, i + 1, :] = y.data[:, :]
    return result

def load_model(file_path, module_name='train'):
    module = importlib.import_module(module_name)
    net = module.Bouncing()
    serializers.load_npz(file_path, net)
    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physical action training')
    parser.add_argument('model_path', type=str, help='Model file path')
    parser.add_argument('-x', type=float, default=5.0, help='X velocity')
    parser.add_argument('-y', type=float, default=5.0, help='Y velocity')
    args = parser.parse_args()

    net = load_model(args.model_path)
    data = make_data(1, 100, vx=args.x, vy=args.y)
    predicted = predict(net, data)
    data = np.concatenate((data, predicted), axis=0)
    plot_animation(data, interval=50)
