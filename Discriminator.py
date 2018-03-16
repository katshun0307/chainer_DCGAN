# -*- coding: utf-8 -*- #

""" Discriminator
"""

from chainer import Chain
import chainer.links as L
import chainer.functions as F
import chainer


class Discriminator(Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        return self.l4(h)
