from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from voxelytics.connect.training.utils import unscale_inputs

logger = getLogger(__name__)


class TensorBoardLoggerPytorch:
    def __init__(
        self,
        tensorboard_dir: Path,
        image_frequency: int = 0,
        device: torch.device = torch.device("cpu"),
    ):

        self.image_frequency = image_frequency

        self.tensorboard_dir = tensorboard_dir

        self.train_writer = SummaryWriter(str(self.tensorboard_dir + "/train"))


    def flush(self) -> None:
        self.train_writer.flush()
        self.valid_writer.flush()

    def _reset_epoch_losses(self) -> None:
        self.total_loss_sum.zero_()

        for training_task_name in self.training_tasks.keys():
            for dataset_name in self.dataset_names:
                self.total_task_dataset_loss_sum[training_task_name][dataset_name][
                    "count"
                ] = 0
                self.total_task_dataset_loss_sum[training_task_name][dataset_name][
                    "loss_sum"
                ].zero_()

        self.total_example_count = 0

    def on_batch(
        self,
        step_within_epoch: int,
        epoch: int,
        global_step: int,
        step_result: "step_result",
        lr: float,
        clips: torch.Tensor,
        masks_enc: torch.Tensor,
        masks_pred: torch.Tensor,
    ) -> None:
        
        self._log_scalars(global_step, step_result, lr)

        if (
            self.image_frequency > 0
            and epoch % self.image_frequency == 0
            and step_within_epoch == 0
        ):

            self._log_clips(global_step, clips)

    
    def _log_scalars(
        self,
        global_step: int,
        step_result: "StepResult",
        lr: float,
    ) -> None:

        self.train_writer.add_scalar(
            "batch_loss", step_result["loss"], global_step
        )
        self.train_writer.add_scalar(
            "batch_loss_jepa", step_result["loss_jepa"], global_step
        )
        self.train_writer.add_scalar(
            "batch_loss_reg", step_result["loss_reg"], global_step
        )
        self.train_writer.add_scalar(
            "learning_rate", lr, global_step,
        )

    def _log_clips(
        self,
        global_step: int,
        clips: torch.Tensor,
    ) -> None:
        # clips have dimensions (batch_size, channels, depth, width, height)

        num_outputs = min(8, clips.shape[0])
        middle_slice = clips.shape[2] // 2
        self.train_writer.add_images(
            "batch_0_0_image",
            clips[:num_outputs, :, middle_slice, :, :],
            global_step=global_step,
            dataformats="NCHW",
        )



