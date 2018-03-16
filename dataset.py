# -*- coding: utf-8 -*- #

""" define datasets
"""

import numpy as np
import chainer
from chainer.dataset import dataset_mixin

class Cifar10Dataset(dataset_mixin.DatasetMixin):

    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, withlabel=False, scale=1.0)
        if test:
            self.ims = d_test
        else:
            self.ims = d_train
        self.ims = self.ims * 2 - 1.0 # [-1.0 ~ 1.0]
        print("loaded cifar-10. shape is ", self.ims.shape)

    def __len__(self):
        return self.ims.shape[0]

    def get_example(self, i):
        return self.ims[i]


if __name__ == '__main__':
    data = Cifar10Dataset(False)
    data.get_example(10)
    # print(len(data))