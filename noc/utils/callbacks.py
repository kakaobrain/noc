# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

from pytorch_lightning.callbacks import Callback


class CallbackTrainer(Callback):
    def __init__(self, dataloader=None):
        super().__init__()

        self._dataloader = dataloader

    def on_train_epoch_start(self, trainer, pl_module):
        """
        We need to set the current epoch to train dataset, because iterable dataset (e.g., webdataset)
        requires the current epoch to shuffle filenames properly.
        Otherwise, we will get the same sequence in every epoch
        """
        self._dataloader.dataset.set_epoch(trainer.current_epoch)
