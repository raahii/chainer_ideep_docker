from glob import glob
import argparse
import os

import numpy as np
import chainer
import imageio
from chainer.backends.intel64 import is_ideep_available

import googlenet


def load_label():
    with open("label.txt", 'r') as f:
        lines = f.readlines()
    return list(map(lambda x: x.rstrip(), lines))


def find_max_epoch(model_dir="result"):
    models = glob(os.path.join(model_dir, 'model_epoch_*.npz'))
    max_epoch = 0
    for model in models:
        name = os.path.basename(model)
        name = os.path.splitext(name)[0]
        epoch = int(name.split("_")[-1])
        if epoch > max_epoch:
            max_epoch = epoch

    return max_epoch


def predict(enable_ideep):
    labels = load_label()
    model_dir = "result"
    epoch = find_max_epoch(model_dir)
    model = googlenet.GoogLeNet()
    mean = np.load("mean.npy")

    print("loading...", epoch)
    chainer.serializers.load_npz(os.path.join(
        model_dir, 'model_epoch_{}.npz'.format(epoch)), model)

    if enable_ideep:
        model.to_intel64()

    paths = glob("reshaped/buddha/image_*.jpg")
    accuracy_cnt = 0
    for img_path in paths:
        image = imageio.imread(img_path)
        image = image.transpose(2, 0, 1).astype(np.float32)
        _, h, w = image.shape
        crop_size = model.insize
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]

        if enable_ideep:
            mode = "always"
        else:
            mode = "never"
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False), chainer.using_config('use_ideep', mode):
            y = model.predict(np.array([image]))
        idx = np.argmax(y.data[0])
        if idx == 0:
            accuracy_cnt += 1
    print("total accuracy rate = ", accuracy_cnt / len(paths))


def main():
    # enable_ideep = is_ideep_available()
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_ideep', action='store_true')
    args = parser.parse_args()
    predict(args.enable_ideep)


if __name__ == '__main__':
    main()
