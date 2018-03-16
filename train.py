# -*- coding: utf-8 -*- #

""" train network
"""

import sys, os
sys.path.append(os.pardir)
from dcgan.dataset import Cifar10Dataset
from dcgan.Updater import Updater
from dcgan.Discriminator import Discriminator
from dcgan.Generator import Generator
from chainer import training
from chainer.training import extensions
import chainer

# setup dataset
batch_size = 128
train_dataset = Cifar10Dataset()
train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size)

# setup models
models = []
opts = {}
updater_args = {
    'iterator': {'main': train_iter},
}

generator = Generator()
discriminator = Discriminator()
models.append(generator)
models.append(discriminator)

# setup optimizers
def make_optimizer(model):
    optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.0, beta2=0.9)
    optimizer.setup(model)
    return optimizer

opts['opt_gen'] = make_optimizer(generator)
opts['opt_dis'] = make_optimizer(discriminator)

updater_args['optimizer'] = opts
updater_args['models'] = models

# setup updater and trainer
max_iter = 2
updater = Updater(**updater_args)
trainer = training.Trainer(updater, (max_iter, 'iteration'), out='result')

# setup logging
report_keys = ['loss_dis', 'loss_gen', 'inception_mean', 'inception_std', 'FID']
display_interval = 100
for m in models: # for both generator and discriminator
    trainer.extend(extensions.LogReport(keys=report_keys, trigger=(display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(display_interval, 'iteration'))

trainer.run()
