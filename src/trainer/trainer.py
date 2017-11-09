import heapq
from torch.utils.trainer.trainer import Trainer
import torch as torch
from torch.utils.trainer.plugins import *
from trainer.plugins.saverplugin import SaverPlugin
from trainer.plugins.validationplugin import ValidationPlugin

class BreedsTrainer(object):
    def __init__(self, model, loader, criterion, optimizer):
        self.loader = loader
        self.trainer = Trainer(model=model,
                               dataset=self.loader.get_train_loader(),
                               criterion=criterion,
                               optimizer=optimizer)
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
