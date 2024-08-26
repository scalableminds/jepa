# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# try:
#     # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
#     # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
#     # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
#     # --          TO EACH PROCESS
#     os.environ["CUDA_VISIBLE_DEVICES"] = 1
# except Exception:
#     pass
import logging
import os
import pprint

import numpy as np
import src.models.vision_transformer as vit
import torch
import torch.multiprocessing as mp
from src.datasets.data_manager import (
    init_data,
)
from src.models.utils.label_patchwise_embed import LabelPatchwiseEmbed3D
from src.utils.distributed import init_distributed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get("pretrain")
    checkpoint_key = args_pretrain.get("checkpoint_key", "target_encoder")
    model_name = args_pretrain.get("model_name", None)
    patch_size = args_pretrain.get("patch_size", None)
    pretrain_folder = args_pretrain.get("folder", None)
    ckp_fname = args_pretrain.get("checkpoint", None)
    tag = args_pretrain.get("write_tag", None)
    use_sdpa = args_pretrain.get("use_sdpa", True)
    use_SiLU = args_pretrain.get("use_silu", False)
    tight_SiLU = args_pretrain.get("tight_silu", True)
    uniform_power = args_pretrain.get("uniform_power", False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get("tubelet_size", 2)
    pretrain_frames_per_clip = args_pretrain.get("frames_per_clip", 1)

    # -- DATA
    args_data = args_eval.get("data")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    eval_num_segments = args_data.get("num_segments", 1)
    eval_frames_per_clip = args_data.get("frames_per_clip", 16)
    eval_frame_step = args_pretrain.get("frame_step", 4)
    eval_duration = args_pretrain.get("clip_duration", None)
    eval_num_views_per_segment = args_data.get("num_views_per_segment", 1)

    # -- OPTIMIZATION
    args_opt = args_eval.get("optimization")
    resolution = args_opt.get("resolution", 224)
    batch_size = 1
    use_bfloat16 = args_opt.get("use_bfloat16")

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "embeddings_classification/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")

    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
    )
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dataloader = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_segments=1,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
    )

    logger.info("Dataloader created...")

    # TODO: make foreground id and fraction adaptable
    label_patchwise_layer = LabelPatchwiseEmbed3D(
        patch_size=patch_size, tubelet_size=tubelet_size
    )
    predict_embeddings(
        device, encoder, dataloader, label_patchwise_layer, use_bfloat16, folder
    )


def predict_embeddings(
    device, encoder, data_loader, label_patchwise_layer, use_bfloat16, folder
):
    for itr, data in enumerate(data_loader):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            clips = torch.cat([d.to(device, non_blocking=True) for d in data[0]], dim=0)
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)

            # Forward
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
            labels_patchwise = label_patchwise_layer(labels)

            # store embeddings and labels
            np.save(
                f"{folder}batch_{itr}_embeddings.npy", outputs.cpu().detach().numpy()
            )
            np.save(
                f"{folder}batch_{itr}_labels.npy",
                labels_patchwise.cpu().detach().numpy(),
            )


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = torch.load(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(
        f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}'
    )
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="WkWDataset",
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
):
    # # Make Video Transforms
    # transform = make_transforms(
    #     training=training,
    #     num_views_per_clip=num_views_per_segment,
    #     random_horizontal_flip=False,
    #     random_resize_aspect_ratio=(0.75, 4 / 3),
    #     random_resize_scale=(0.08, 1.0),
    #     reprob=0.25,
    #     auto_augment=True,
    #     motion_shift=False,
    #     crop_size=resolution,
    # )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=None,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
    )
    return data_loader


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key="target_encoder",
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(
        encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key
    )
    return encoder
