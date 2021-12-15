from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class MyTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        self.model = nn.Sequential(
            nn.Linear(128, 10),
            nn.LogSoftmax(),
        )
        self.model = self.context.wrap_model(self.model)
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.context.get_hparam("learning_rate"))
        self.optimizer = self.context.wrap_optimizer(self.optimizer)

        self.batch_size = self.context.get_per_slot_batch_size()

    def build_training_data_loader(self) -> DataLoader:
        # Fill

    def build_validation_data_loader(self) -> DataLoader:
        # Fill


    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        output = self.model(data)
        loss = torch.nn.functional.nll_loss(output, labels)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss}


    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        data, labels = batch

        output = self.model(data)
        loss = torch.nn.functional.nll_loss(output, labels)

        return {"val_loss": loss}

