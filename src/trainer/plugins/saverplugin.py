import torch
from glob import glob
import os
from torch.utils.trainer.plugins.plugin import Plugin

class SaverPlugin(Plugin):

    last_pattern = 'ep{}-it{}'
    best_pattern = 'best-ep{}-it{}'

    def __init__(self, checkpoints_path, keep_old_checkpoints):
        super().__init__([(1, 'epoch')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
        torch.save(
            self.trainer.model.state_dict(),
            os.path.join(
                self.checkpoints_path,
                self.last_pattern.format(epoch_index, self.trainer.iterations)
            )
        )

        cur_val_loss = self.trainer.stats['validation_loss']['last']
        if cur_val_loss < self._best_val_loss:
            self._clear(self.best_pattern.format('*', '*'))
            torch.save(
                self.trainer.model.state_dict(),
                os.path.join(
                    self.checkpoints_path,
                    self.best_pattern.format(
                        epoch_index, self.trainer.iterations
                    )
                )
            )
            self._best_val_loss = cur_val_loss

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)