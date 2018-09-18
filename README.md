# chainer ideep docker

```
docker build . -t chainer_ideep
docker run -it chainer_ideep
```

# ImageNet example

On MacBook Pro 2016 (3.3 GHz Intel Core i7)

```
(base) root@c3bfc92c36cc:/chainer_ideep/chainer-imagenet# time python predict.py
loading... 46
total accuracy rate =  0.011764705882352941

real    0m25.180s
user    0m48.040s
sys     0m1.160s
```
```
(base) root@c3bfc92c36cc:/chainer_ideep/chainer-imagenet# time python predict.py --enable_ideep
loading... 46
total accuracy rate =  0.011764705882352941

real    0m5.237s
user    0m8.910s
sys     0m0.730s
(base) root@c3bfc92c36cc:/chainer_ideep/chainer-imagenet#
```

# reference

- [Chainer 4.0 + iDeep がすごい](https://qiita.com/ikeyasu/items/62781ac2eeff5be4e5f4)