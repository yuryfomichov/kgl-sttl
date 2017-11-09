import heapq
from torch.utils.trainer.trainer import Trainer
import torch as torch
from torch.utils.trainer.plugins import *
from trainer.plugins.saverplugin import SaverPlugin
from trainer.plugins.validationplugin import ValidationPlugin
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class BreedsTrainer(object):
    def __init__(self, model, loader, criterion, optimizer, data_type):
        self.data_type = data_type
        self.loader = loader
        self.trainer = Trainer(model=model,
                               dataset=self.loader.get_train_loader(),
                               criterion=criterion,
                               optimizer=optimizer)
        self.model = model
        self.trainer.cuda = True if torch.cuda.is_available() else False
        self._register_plugins()

    def _register_plugins(self):
        self.trainer.register_plugin(AccuracyMonitor())
        self.trainer.register_plugin(LossMonitor())
        self.trainer.register_plugin(ProgressMonitor())
        self.trainer.register_plugin(TimeMonitor())
        self.trainer.register_plugin(ValidationPlugin(self.loader.get_val_loader(), self.loader.get_test_loader()))
        self.trainer.register_plugin(SaverPlugin('checkpoints/', False))
        self.trainer.register_plugin(Logger(['accuracy', 'loss', 'progress', 'time', 'validation_loss', 'test_loss']))

    def run(self, lrs=[1e-02], epochs=[10]):
        for q in self.trainer.plugin_queues.values():
            heapq.heapify(q)

        count = 0
        for (lr, epoch) in zip(lrs, epochs):
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = lr
            print('lr is set to:' + str(lr))
            print('train for epoch: ' + str(epoch))
            for i in range(1, epoch + 1):
                self.trainer.train()
                self.trainer.call_plugins('epoch', count + i)
            count += epoch
            self.pseudo_labling()

    def pseudo_labling(self):
        batches = 2
        i = 0
        self.model.eval()
        for x, y in self.loader.get_submission_loader():
            _, result = nn.Softmax()(self.model(Variable(x.type(self.data_type), volatile=True))).data.cpu().max(1)
            self.loader.train_data = (self.loader.train_data[0].append(self.loader.submission_data[0].loc[y]), np.hstack((self.loader.train_data[1], result.numpy())))
            i = i+1
            if i == batches:
                break
        self.trainer.dataset = self.loader.get_train_loader()
        self.model.train()
