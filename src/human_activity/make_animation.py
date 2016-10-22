import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import six

def update(index, scat, items):
    item = items[index]
    scat.set_offsets(item[:, :2])
    scat.set_3d_properties(item[:, 2], 'z')
    return scat

def plot_3d_animation(data, step=1, interval=100):
    sub_data = np.ascontiguousarray(data[::step,:,:])
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([-2000, 2000])
    ax.set_ylim3d([-2000, 2000])
    ax.set_zlim3d([-1000, 2000])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Human Activity')

    scat = ax.scatter(sub_data[0, :, 0], sub_data[0, :, 1], sub_data[0, :, 2], c=['g', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b'], s=50, alpha=1)

    # Creating the Animation object
    anim = animation.FuncAnimation(
        fig, update, len(sub_data), fargs=(scat, sub_data),
        interval=interval, blit=False
    )
    plt.show()

def preprocess(x):
    n, m, ch = x.shape
    h = x.reshape((n, -1, 3))
    # subtract head x,y positions and constant z position
    base = np.mean(h, axis=(0, 1), keepdims=True)
    base[:, :, 2] = 800
    h = h - base
    # calculate body direction
    lankle_x = np.mean(h[:, 6, 0], axis=0, keepdims=True)
    lankle_y = np.mean(h[:, 6, 1], axis=0, keepdims=True)
    rankle_x = np.mean(h[:, 8, 0], axis=0, keepdims=True)
    rankle_y = np.mean(h[:, 8, 1], axis=0, keepdims=True)
    angle = np.arctan2(lankle_y - rankle_y, lankle_x - rankle_x)
    angle = angle.reshape((-1,))
    cos = np.cos(angle)
    sin = np.sin(angle)
    rot = np.asarray([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]], dtype=np.float32)
    return np.dot(h, rot).reshape(x.shape)

def read_file(file_path):
    data = np.loadtxt(file_path, usecols=six.moves.range(1, 28), dtype=np.float32)
    row_num, col_num = data.shape
    return data.reshape((row_num, -1, 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physical action training')
    parser.add_argument('file_path', type=str, help='data file path')
    args = parser.parse_args()

    data = read_file(args.file_path)
    data = preprocess(data)
    plot_3d_animation(data, step=8, interval=40)
