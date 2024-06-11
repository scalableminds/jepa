from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

from voxelytics.connect.training.utils import unscale_inputs

logger = getLogger(__name__)


class TensorBoardLoggerPytorch:
    def __init__(
        self,
        tensorboard_dir: Path,
        tag: str,
        image_frequency: int = 0,
        device: torch.device = torch.device("cpu"),
    ):

        self.image_frequency = image_frequency

        self.tensorboard_dir = tensorboard_dir

        self.train_writer = SummaryWriter(str(self.tensorboard_dir + "/train_" + tag))


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
        h_raw: torch.Tensor,
    ) -> None:
        
        self._log_scalars(global_step, step_result, lr)

        if (
            self.image_frequency > 0
            and epoch % self.image_frequency == 0
            and step_within_epoch == 0
        ):
            self._log_clips(global_step, clips)
            self._log_mask(clips, global_step, masks_pred)
            self._log_prediction_clustered(global_step, clips, h_raw)
            self._log_mito_classifier()

    
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
        if step_result["grad_stats_pred"] is not None:
            self.train_writer.add_scalar(
                "grad_stats_pred_first_layer", step_result["grad_stats_pred"].first_layer, global_step
            )
            self.train_writer.add_scalar(
                "grad_stats_pred_last_layer", step_result["grad_stats_pred"].last_layer, global_step
            )
            self.train_writer.add_scalar(
                "grad_stats_pred_global_norm", step_result["grad_stats_pred"].global_norm, global_step
            )
        if step_result["grad_stats"] is not None:
            self.train_writer.add_scalar(
                "grad_stats_first_layer", step_result["grad_stats"].first_layer, global_step
            )
            self.train_writer.add_scalar(
                "grad_stats_last_layer", step_result["grad_stats"].last_layer, global_step
            )
            self.train_writer.add_scalar(
                "grad_stats_global_norm", step_result["grad_stats"].global_norm, global_step
            )
        if step_result["optim_stats"] is not None:
            self.train_writer.add_scalar(
                "optim_stats_exp_avg_avg", step_result["optim_stats"].get("exp_avg").avg, global_step
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
            "batch_0_images",
            clips[:num_outputs, :, middle_slice, :, :],
            global_step=global_step,
            dataformats="NCHW",
        )

    def _log_mask(
        self,
        clips: torch.Tensor,
        global_step: int,
        masks_pred: torch.Tensor,
    ) -> None:

        batch_size = clips.shape[0]

        reshaped_clip = clips.reshape(batch_size,8,2,14,16,14,16).permute(0,3,5,1,4,6,2) #(BxHxWxDxTHxTWxTD)

        mask_tensor = np.ones((batch_size, 1568), dtype=int)

        for i in range(mask_tensor.shape[0]):
            mask_tensor[i, masks_pred[0][i].cpu()] = 0

        mask = torch.from_numpy(mask_tensor).reshape(batch_size,8,14,14,1,1,1).permute(0,2,3,1,4,5,6)

        masked_clips = torch.mul(reshaped_clip.cpu(), mask.cpu()).permute(0,3,6,1,4,2,5).reshape(batch_size,1,16,224,224)
        
        num_outputs = min(8, clips.shape[0])

        self.train_writer.add_images(
            "batch_0_images_masked",
            masked_clips[:num_outputs, :, 0, :, :],
            global_step=global_step,
            dataformats="NCHW",
        )

    def _log_prediction_clustered(
        self,
        global_step: int,
        clips: torch.Tensor,
        predictions_raw = torch.Tensor,
    ) -> None:
        batch_size = clips.shape[0]

        #flatten predictions_raw from batch_size x 16*16*2 x embedded_dim to batch_size*16*16*2 x embedded_dim
        predictions_raw_flattened = predictions_raw.reshape(-1, predictions_raw.shape[-1]).cpu().detach().numpy()

        #perform kmeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42).fit(predictions_raw_flattened)
        labels = kmeans.labels_
        
        #reshape and upsample
        labels = labels.reshape(8,14,14,8)
        upsampled_labels = np.repeat(np.repeat(np.repeat(labels, 16, axis=1), 16, axis=2), 2, axis=3)

        #reorder dimensions to match clips and add a new dimension for the channel
        upsampled_labels = torch.from_numpy(upsampled_labels).permute(0,3,1,2).unsqueeze(1)

        num_outputs = min(8, clips.shape[0])

        blended = torch.zeros((num_outputs, 380, 400, 3))

        for i in range(num_outputs):
            fig = plt.figure(frameon=False)
            plt.axis('off')
            plt.imshow(clips[i,0,8,:,:].cpu().detach().numpy(), cmap='gray')
            plt.imshow(upsampled_labels[i,0,8,:,:].cpu().detach().numpy(), cmap='OrRd', alpha=0.3, interpolation='none')
            fig.canvas.draw()
            rgba_image = np.array(fig.canvas.renderer.buffer_rgba())
            background = np.ones_like(rgba_image[:, :, :3]) * 255
            rgb_image = (1 - rgba_image[:, :, 3:4] / 255.0) * background + (rgba_image[:, :, :3] / 255.0) * (rgba_image[:, :, 3:4] / 255.0)
            rgb_image = rgb_image[50:-50, 120:-120, :]
            blended[i] = torch.tensor(rgb_image)
            plt.close(fig)

        self.train_writer.add_images(
            "batch_0_images_clustered",
            blended,
            global_step=global_step,
            dataformats="NHWC",
        )

    def _log_mito_classifier(
        self,
        ):
        pass




 
        