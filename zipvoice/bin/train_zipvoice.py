#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang,
#                                                       Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script trains a ZipVoice model with the flow-matching loss.

Usage:

python3 -m zipvoice.bin.train_zipvoice \
    --world-size 8 \
    --use-fp16 1 \
    --num-epochs 11 \
    --max-duration 500 \
    --lr-hours 30000 \
    --model-config conf/zipvoice_base.json \
    --tokenizer emilia \
    --token-file "data/tokens_emilia.txt" \
    --dataset emilia \
    --manifest-dir data/fbank \
    --exp-dir exp/zipvoice
"""

import argparse
import copy
import json
import logging
import os
import random
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut, CutSet
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import zipvoice.utils.diagnostics as diagnostics
from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.lm_tokenizer import (
    LanguageModelTokenizer,
    normalize_language_name,
    sample_lang_tag,
)
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import (
    load_checkpoint,
    remove_checkpoints,
    resume_checkpoint,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from zipvoice.utils.common import (
    AttributeDict,
    GradScaler,
    MetricsTracker,
    cleanup_dist,
    create_grad_scaler,
    get_adjusted_batch_count,
    get_env_info,
    get_parameter_groups_with_lrs,
    prepare_input,
    set_batch_count,
    setup_dist,
    setup_logger,
    str2bool,
    torch_autocast,
)
from zipvoice.utils.hooks import register_inf_check_hooks
from zipvoice.utils.lr_scheduler import Eden, FixedLRScheduler, LRScheduler
from zipvoice.utils.optim import ScaledAdam

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12356,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=11,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--num-iters",
        type=int,
        default=0,
        help="Number of iter to train, will ignore num_epochs if > 0.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="""Checkpoints of pre-trained models, will load it if not None
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/zipvoice",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument("--warmup-batches", type=int, default=500)
    parser.add_argument(
        "--base-lr", type=float, default=0.02, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--lr-hours",
        type=float,
        default=0,
        help="""If positive, --epoch is ignored and it specifies the number of hours
        that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=50,
        help="""Reference batch duration for purposes of adjusting batch counts for"
        setting various schedules inside the model".
        """,
    )

    parser.add_argument(
        "--finetune",
        type=str2bool,
        default=False,
        help="Whether to use the fine-tuning mode, will used a fixed learning rate "
        "schedule and skip the large dropout phase.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--scan-oom",
        type=str2bool,
        default=False,
        help="Scan pessimistic batches to see whether they cause OOMs.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=5000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--valid-by-epoch",
        type=str2bool,
        default=False,
        help="""Whether to validate after each epoch. If False, will validate 
        after every save_every_n iterations.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--condition-drop-ratio",
        type=float,
        default=0.2,
        help="The drop rate of text condition during training.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "custom"],
        help="The used training dataset",
    )

    parser.add_argument(
        "--train-manifest",
        type=str,
        help="Path of the training manifest",
    )

    parser.add_argument(
        "--dev-manifest",
        type=str,
        help="Path of the validation manifest",
    )

    parser.add_argument(
        "--min-len",
        type=float,
        default=1.0,
        help="The minimum audio length used for training",
    )

    parser.add_argument(
        "--max-len",
        type=float,
        default=30.0,
        help="The maximum audio length used for training",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/zipvoice_base.json",
        help="The model configuration file.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple", "multilingual", "fusion"],
        help="Tokenizer type. 'multilingual' wraps an existing pretrained "
        "HuggingFace tokenizer (see --pretrained-tokenizer-name) instead of "
        "phonemizing text -- see docs/adr/2026-08-28__choose_qwen25_tokenizer.md. "
        "'fusion' is the Qwen+phoneme fusion frontend: phones (via "
        "FusionTokenizer, per --token-file) fused with a truncated Qwen's "
        "per-group features -- see "
        "docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md.",
    )

    parser.add_argument(
        "--lm-layers",
        type=int,
        default=4,
        help="Number of Qwen transformer layers to run for the fusion "
        "frontend's Qwen branch (N in the ADR). 4 was validated empirically "
        "with a linear probe; 2/4/8/12 all scored ~96-98% and 24 was worse. "
        "Only used when --tokenizer=fusion.",
    )

    parser.add_argument(
        "--gate-init-eps",
        type=float,
        default=0.01,
        help="Initial Qwen mixing weight for the fusion gate -- a "
        "near-phone-only start, with no floor (gate collapse is an accepted, "
        "monitored risk per the ADR). Only used when --tokenizer=fusion.",
    )

    parser.add_argument(
        "--pretrained-tokenizer-name",
        type=str,
        default=None,
        help="HuggingFace model id whose tokenizer to wrap, when "
        "--tokenizer=multilingual. Defaults to LanguageModelTokenizer's own "
        "default (Qwen2.5-0.5B) if not given.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )

    parser.add_argument(
        "--lang-auto-prob",
        type=float,
        default=0.05,
        help="When --tokenizer=multilingual, probability of replacing an "
        "utterance's ground-truth [LANG:xx] tag with [LANG:auto], so the "
        "model still works when a caller doesn't specify a language at "
        "inference. The ground-truth language is read per-utterance from "
        "each cut's own supervision.language field (e.g. the "
        "'language': 'Vietnamese' field in the user's YouTube-corpus schema) "
        "-- NOT a single flag for the whole dataset, since a training run "
        "may mix corpora in different languages. Utterances with no "
        "recognized language field get no [LANG:xx] tag at all. Applied "
        "fresh per epoch (lhotse cuts are lazy), matching OmniVoice's "
        "label-dropout approach. See "
        "docs/proposals/2026-08-28__primary_language_conditioning.md.",
    )

    parser.add_argument(
        "--lang-wrong-prob",
        type=float,
        default=0.0,
        help="When --tokenizer=multilingual, probability of using a "
        "deliberately incorrect [LANG:xx] tag for a given utterance -- for "
        "the true/auto/wrong probe described in "
        "docs/proposals/2026-08-28__primary_language_conditioning.md. "
        "Defaults to 0 (off); --lang-auto-prob + --lang-wrong-prob must be "
        "<= 1.",
    )

    parser.add_argument(
        "--token-file",
        type=str,
        default="data/tokens_emilia.txt",
        help="The file that contains information that maps tokens to ids,"
        "which is a text file with '{token}\t{token_id}' per line.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - env_info:  A dict containing information about the environment.

    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "env_info": get_env_info(),
        }
    )

    return params


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
    zero_duration_mask: Optional[List[List[bool]]] = None,
    fusion_fields: Optional[dict] = None,
    lm_extractor=None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      features:
        The target acoustic feature.
      features_lens:
        The number of frames of each utterance.
      tokens:
        Input tokens that representing the transcripts.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      fusion_fields:
        The fusion frontend's per-utterance extras (`phone_groups`,
        `lm_token_ids`, `lm_token_groups`) from `prepare_input`, or None for
        every other tokenizer.
      lm_extractor:
        A `TruncatedQwenExtractor`, required when `fusion_fields` is given.
        Runs live here (per the ADR's point 6: no offline cache in v1 --
        Qwen2.5-0.5B at 4 layers is cheap enough that a cache's
        invalidation/serialization complexity isn't justified yet). It is
        frozen and runs under `no_grad`, so it costs a forward pass, not a
        backward one.
    """

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape

    noise = torch.randn_like(features)  # (B, T, F)

    # Sampling t from uniform distribution
    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (
            (torch.arange(batch_size, device=device) / batch_size)
            .unsqueeze(1)
            .unsqueeze(2)
        )
    model_fusion_kwargs = {}
    if fusion_fields is None:
        # The converse of the assert below, and the more dangerous direction.
        # A configured fusion run whose batch carries no fusion fields looks
        # *identical* to an intentional phone-only run: the model falls back
        # to the phone embedding and trains happily. That is how a whole
        # training run was spent ~93% phone-only without anything failing
        # (see the sentence-boundary bug in `_split_into_groups`). Missing
        # metadata on a fusion run is a pipeline error, not an alignment
        # fallback, so it must not be absorbed silently.
        assert lm_extractor is None, (
            "this run is configured for the fusion text frontend, but this "
            "batch carries no fusion fields at all -- phone_groups never "
            "reached the collator. That is a pipeline error, not the "
            "per-utterance alignment fallback (which keeps the fields and "
            "sets lm_token_groups=None for the affected utterances)."
        )
    else:
        assert lm_extractor is not None, (
            "fusion_fields require a lm_extractor to turn lm_tokens into "
            "per-group Qwen vectors"
        )
        lm_group_features, lm_group_mask = lm_extractor.pooled_groups(
            fusion_fields["lm_token_ids"], fusion_fields["lm_token_groups"]
        )
        model_fusion_kwargs = {
            "phone_groups": fusion_fields["phone_groups"],
            "lm_group_features": lm_group_features.to(device),
            "lm_group_mask": lm_group_mask.to(device),
        }
        # Reason-specific coverage, so silent conditioning loss is visible in
        # the training log instead of having to be discovered months later by
        # auditing a checkpoint. Utterance-level *and* group-level: an
        # utterance can align yet contribute empty groups that carry no Qwen
        # evidence, which `lm_group_mask` marks false exactly like padding.
        groups_per_utt = fusion_fields["lm_token_groups"]
        n_utt = len(groups_per_utt)
        n_aligned = sum(1 for g in groups_per_utt if g is not None)
        real_groups = sum(len(g) for g in groups_per_utt if g is not None)
        n_valid = int(lm_group_mask.sum().item())
        # Stored pre-multiplied by num_frames, matching how `loss` is stored
        # here: MetricsTracker.norm_items() divides by "frames", so the value
        # that reaches the log is the ratio itself.
        _frames = int(features_lens.sum().item())
        info_coverage = {
            "lm_cov_utt": (n_aligned / n_utt if n_utt else 0.0) * _frames,
            "lm_cov_grp": (
                n_valid / real_groups if real_groups else 0.0
            ) * _frames,
        }

    with torch.set_grad_enabled(is_training):

        loss = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            condition_drop_ratio=params.condition_drop_ratio,
            zero_duration_mask=zero_duration_mask,
            **model_fusion_kwargs,
        )

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames
    if model_fusion_kwargs:
        info.update(info_coverage)

    return loss, info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    lm_extractor=None,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches or every epochs.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer.
      scheduler:
        The learning rate scheduler, we call step() every epoch.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    # Intentional, permanent deviation from master (accepted exception, not
    # drift -- see docs/proposals/2026-08-29__master_backward_compatibility.md):
    # a tqdm progress bar is shown unconditionally, for all tokenizer types,
    # not just multilingual. Requested explicitly by the user this session.
    train_pbar = tqdm(
        train_dl,
        desc=f"Epoch {params.cur_epoch}",
        disable=(rank != 0),
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(train_pbar):
        if batch_idx % 10 == 0:
            if params.finetune:
                set_batch_count(model, get_adjusted_batch_count(params) + 100000)
            else:
                set_batch_count(model, get_adjusted_batch_count(params))

        # `params.batch_idx_train > 0` below is an intentional, permanent
        # deviation from master (accepted exception, not drift -- see
        # docs/proposals/2026-08-29__master_backward_compatibility.md):
        # master validates once at batch 0, before any training step;
        # skipping that was requested explicitly by the user this session,
        # since it wastes compute validating an untrained model.
        if (
            params.valid_by_epoch and batch_idx == 0 and not params.print_diagnostics
        ) or (
            not params.valid_by_epoch
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                lm_extractor=lm_extractor,
            )
            model.train()
            logging.info(
                f"Epoch {params.cur_epoch}, global_batch_idx: {params.batch_idx_train},"
                f" validation: {valid_info}"
            )
            logging.info(
                f"Maximum memory allocated so far is "
                f"{torch.cuda.max_memory_allocated() // 1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        params.batch_idx_train += 1

        batch_size = len(batch["text"])

        (
            tokens,
            zero_duration_mask,
            fusion_fields,
            features,
            features_lens,
        ) = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
            return_zero_duration_mask=True,
            return_fusion_fields=True,
        )

        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    zero_duration_mask=zero_duration_mask,
                    fusion_fields=fusion_fields,
                    lm_extractor=lm_extractor,
                    is_training=True,
                )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            # Use the number of hours of speech to adjust the learning rate
            if params.lr_hours > 0:
                scheduler.step_epoch(
                    params.batch_idx_train
                    * params.max_duration
                    * params.world_size
                    / 3600
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception : {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it. The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have
            # different behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 1024.0 or (
                cur_grad_scale < 4096.0 and params.batch_idx_train % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            train_pbar.set_postfix(
                loss=f"{tot_loss['loss'] / tot_loss['frames']:.4f}", lr=f"{cur_lr:.2e}"
            )

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, "
                f"batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    lm_extractor=None,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        (
            tokens,
            zero_duration_mask,
            fusion_fields,
            features,
            features_lens,
        ) = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
            return_zero_duration_mask=True,
            return_fusion_fields=True,
        )

        loss, loss_info = compute_fbank_loss(
            params=params,
            model=model,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            zero_duration_mask=zero_duration_mask,
            fusion_fields=fusion_fields,
            lm_extractor=lm_extractor,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    features = batch["features"]
    tokens = batch["tokens"]

    logging.info(f"features shape: {features.shape}")
    num_tokens = sum(len(i) for i in tokens)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
    lm_extractor=None,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        (
            tokens,
            zero_duration_mask,
            fusion_fields,
            features,
            features_lens,
        ) = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
            return_zero_duration_mask=True,
            return_fusion_fields=True,
        )
        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):

                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    zero_duration_mask=zero_duration_mask,
                    fusion_fields=fusion_fields,
                    lm_extractor=lm_extractor,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is "
            f"{torch.cuda.max_memory_allocated() // 1000000}MB"
        )


def tokenize_text_fusion(c: Cut, tokenizers: dict):
    """Tokenize one cut with the Qwen+phoneme fusion frontend.

    Unlike the other tokenizers, `FusionTokenizer` is locked to a single
    language per instance (see its docstring), so `tokenizers` maps a
    canonical short code ("en"/"vi"/"zh") to the instance for that language,
    and the cut's own `supervision.language` selects between them.

    That per-utterance language is exactly the "immutable ground truth" the
    ADR's point 1 builds the phone branch on. Sprint 001 found it is *not*
    reliable in this corpus (~95% of "English"-labelled utterances are
    actually Vietnamese with embedded English words), which the author chose
    to leave unfixed -- so this path knowingly phonemizes those utterances
    under the wrong language. Recorded here rather than silently relied on;
    see sprint 003's "Known accepted data-quality issue".

    Attaches the whole artifact's fields to the supervision so the dataset
    can collate them: `tokens` (phone ids -- same field name the rest of the
    pipeline already reads), plus `phone_groups`, `lm_token_ids` and
    `lm_token_groups` for the Qwen branch.
    """
    raw_lang = getattr(c.supervisions[0], "language", None)
    lang = normalize_language_name(raw_lang)
    if lang is None:
        raise ValueError(
            f"cut {c.id!r} has a missing or unrecognized language "
            f"({raw_lang!r}). Every utterance must have a valid language; "
            f"this should have been caught during data preparation (see "
            f"scripts/*/m00_prepare_manifest.py)."
        )
    if lang not in tokenizers:
        raise ValueError(
            f"cut {c.id!r} is {lang!r}, but no FusionTokenizer was built for "
            f"that language (have: {sorted(tokenizers)})."
        )

    artifact = tokenizers[lang].text_to_artifact(c.supervisions[0].text)
    c.supervisions[0].tokens = artifact.phone_ids
    c.supervisions[0].phone_groups = artifact.phone_groups
    c.supervisions[0].lm_token_ids = artifact.lm_token_ids
    c.supervisions[0].lm_token_groups = artifact.lm_token_groups
    # The fusion frontend owns no zero-duration phones: `[LANG:xx]` lives on
    # the lm_token side only, and every phone this emits is acoustic. Cleared
    # rather than merely not-set, so a supervision that arrived carrying a
    # mask from some other tokenizer cannot leak into this batch -- that mask
    # counts lm_tokens, so against phone counts it would either trip
    # `prepare_avg_tokens_durations`' length assert or, if the two counts
    # happened to match, silently zero the duration of real phones.
    # Assigning None is lhotse's documented way to remove a custom field.
    c.supervisions[0].zero_duration_mask = None
    return c


def tokenize_text(
    c: Cut,
    tokenizer,
    lang_auto_prob: float = 0.0,
    lang_wrong_prob: float = 0.0,
    lang_rng: Optional[random.Random] = None,
):
    """
    Args:
      lang_rng: if given (only for LanguageModelTokenizer), the ground-truth
        language is read per-cut from `c.supervisions[0].language` (e.g. the
        "language": "Vietnamese" field in the user's YouTube-corpus schema,
        normalized via `normalize_language_name`) and a `[LANG:xx]` tag is
        prepended to the text before tokenizing, sampled via
        `sample_lang_tag` with the given dropout probabilities. Every cut
        must have a valid, recognized language -- this is expected to be
        enforced already during data preparation (see
        scripts/*/m00_prepare_manifest.py), and is re-checked here as a
        safety net. Called fresh on every invocation -- since train/dev
        cuts are lazy lhotse CutSets, this function runs again each epoch,
        so the sampled tag differs run to run rather than being fixed once
        at manifest-prep time.
    """
    if hasattr(c.supervisions[0], "tokens"):
        existing = c.supervisions[0].tokens
        # This branch means "the manifest was pre-tokenized into symbol
        # *strings*". `tokens_to_token_ids` looks each one up in a str-keyed
        # vocabulary and silently `continue`s past anything missing, so a
        # sequence of ints -- which is what the fusion frontend stores in
        # this same field -- would not raise here: every id would miss, and
        # the cut would come out with `tokens=[]`, failing much later and
        # much less obviously in duration allocation.
        if any(isinstance(t, int) for t in existing):
            raise ValueError(
                f"cut {c.id!r} already carries integer `tokens`, which is "
                f"what the fusion frontend writes. This tokenizer expects "
                f"pre-tokenized input to be symbol strings; re-run "
                f"tokenization from text instead of reusing a fused cut."
            )
        tokens = tokenizer.tokens_to_token_ids([existing])
    else:
        text = c.supervisions[0].text
        if lang_rng is not None:
            raw_lang = getattr(c.supervisions[0], "language", None)
            true_lang = normalize_language_name(raw_lang)
            if true_lang is None:
                raise ValueError(
                    f"cut {c.id!r} has a missing or unrecognized language "
                    f"({raw_lang!r}). Every utterance must have a valid "
                    f"language; this should have been caught during data "
                    f"preparation (see scripts/*/m00_prepare_manifest.py)."
                )
            tag = sample_lang_tag(true_lang, lang_auto_prob, lang_wrong_prob, lang_rng)
            text = f"{tag} {text}"
        tokens = tokenizer.texts_to_token_ids([text])
    c.supervisions[0].tokens = tokens[0]
    if hasattr(tokenizer, "zero_duration_mask"):
        c.supervisions[0].zero_duration_mask = tokenizer.zero_duration_mask(tokens[0])
    else:
        c.supervisions[0].zero_duration_mask = None
    # Symmetrically to `tokenize_text_fusion`: the non-fusion frontends emit
    # no groups, so clear any that a fusion pass left behind. Otherwise the
    # dataset would collate fusion fields for a run whose `lm_extractor` is
    # None, and `compute_fbank_loss` would reject the batch.
    c.supervisions[0].phone_groups = None
    c.supervisions[0].lm_token_ids = None
    c.supervisions[0].lm_token_groups = None
    return c


#: Settings that define what the text actually looks like to the model. A
#: resume that changes one of these loads without error and then trains
#: against a different representation than the checkpoint learned.
RESUME_CRITICAL_KEYS = (
    "lm_layers",
    "tokenizer",
    "text_frontend",
    # Selects *which* frozen Qwen supplies the conditioning. A different
    # checkpoint at the same hidden width loads and runs without complaint.
    "pretrained_tokenizer_name",
)


def _effective_setting(key: str, value):
    """What a saved/current setting actually selects.

    `pretrained_tokenizer_name=None` is a real value, not a missing one: the
    extractor resolves it as `params.pretrained_tokenizer_name or
    DEFAULT_MODEL_NAME`. So None and the explicit default name select the
    same frozen Qwen and must compare equal, or resuming a default-model run
    while passing the name explicitly would be rejected for no reason.
    """
    from zipvoice.models.modules.lm_extractor import DEFAULT_MODEL_NAME

    if key == "pretrained_tokenizer_name":
        return value or DEFAULT_MODEL_NAME
    return value


def check_resume_text_frontend(params, checkpoints) -> None:
    """Refuse to resume a fusion run whose text frontend has changed.

    `resume_checkpoint` restores only the five training counters, so every
    other setting silently comes from this invocation's CLI rather than the
    checkpoint. For most settings that is intended -- you may well want a
    different learning rate on resume. For the frozen extractor's depth it
    is not: `lm_layers` changes which Qwen layer the conditioning comes
    from, but *not* the pooled width (always 896), so
    `--tokenizer fusion --start-epoch 2` with `--lm-layers` omitted loads
    cleanly, silently falls back to the argparse default, and continues
    training against a different representation with nothing to indicate it.
    The same argument applies to `pretrained_tokenizer_name`, which selects
    *which* frozen Qwen is used: a different checkpoint at the same hidden
    width also loads without complaint.
    """
    if checkpoints is None or params.tokenizer != "fusion":
        return
    for key in RESUME_CRITICAL_KEYS:
        # `key not in checkpoints` is the test for "this checkpoint predates
        # the setting", NOT `saved is None`: None is a legitimate *value* for
        # `pretrained_tokenizer_name`, meaning the default model. Treating it
        # as missing metadata would wave through a resume that swapped in a
        # different same-width Qwen.
        if key not in params or key not in checkpoints:
            continue
        saved = _effective_setting(key, checkpoints[key])
        current = _effective_setting(key, params[key])
        if saved != current:
            raise ValueError(
                f"resuming with {key}={current!r} but the checkpoint was "
                f"trained with {key}={saved!r}. The model would load without "
                f"error and train against a different text representation -- "
                f"pass the checkpoint's value explicitly if this is deliberate."
            )


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    params.valid_interval = params.save_every_n
    # Set epoch to a large number to ignore it.
    if params.num_iters > 0:
        params.num_epochs = 1000000
    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    params.update(model_config["model"])
    params.update(model_config["feature"])

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    if params.tokenizer != "multilingual":
        # LanguageModelTokenizer has no local tokens.txt -- its vocabulary
        # (including the [LANG:xx] tokens added on top of the pretrained
        # base) is saved separately below, once it's actually constructed.
        copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    if torch.cuda.is_available():
        params.device = torch.device("cuda", rank)
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    # Populated only by the fusion frontend; every other tokenizer leaves
    # these None and the fusion-specific code paths stay inert.
    fusion_tokenizers = None
    lm_extractor = None

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=params.token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=params.token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=params.token_file, lang=params.lang)
    elif params.tokenizer == "fusion":
        # One FusionTokenizer per language (each instance is locked to a
        # single `lang`), sharing one LanguageModelTokenizer so the Qwen
        # tokenizer is loaded once. See the fusion ADR.
        from zipvoice.models.modules.lm_extractor import (
            DEFAULT_MODEL_NAME,
            TruncatedQwenExtractor,
        )
        from zipvoice.tokenizer.fusion_tokenizer import FusionTokenizer

        lm_tokenizer_kwargs = {}
        if params.pretrained_tokenizer_name is not None:
            lm_tokenizer_kwargs["pretrained_model_name"] = (
                params.pretrained_tokenizer_name
            )
        lm_tokenizer = LanguageModelTokenizer(**lm_tokenizer_kwargs)
        fusion_tokenizers = {
            lang: FusionTokenizer(
                token_file=params.token_file, lang=lang, lm_tokenizer=lm_tokenizer
            )
            for lang in ("en", "vi", "zh")
        }
        # `tokenizer` still stands in for vocab_size/pad_id below: they are
        # the *phone* vocabulary's, identical across the three instances.
        tokenizer = fusion_tokenizers["vi"]
        lm_extractor = TruncatedQwenExtractor(
            model_name=params.pretrained_tokenizer_name or DEFAULT_MODEL_NAME,
            num_layers=params.lm_layers,
            device=params.device,
        )
        # Same reason the multilingual path saves its tokenizer: inference
        # must rebuild the identical lm_token vocabulary.
        lm_tokenizer.hf_tokenizer.save_pretrained(f"{params.exp_dir}/tokenizer")
    elif params.tokenizer == "multilingual":
        multilingual_kwargs = {}
        if params.pretrained_tokenizer_name is not None:
            multilingual_kwargs["pretrained_model_name"] = (
                params.pretrained_tokenizer_name
            )
        tokenizer = LanguageModelTokenizer(**multilingual_kwargs)
        # Save the exact tokenizer (incl. the [LANG:xx] tokens added on top
        # of the pretrained base) so inference can reload the same vocabulary
        # -- there is no local tokens.txt to copy for this tokenizer type.
        tokenizer.hf_tokenizer.save_pretrained(f"{params.exp_dir}/tokenizer")
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=params.token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    if params.tokenizer == "fusion":
        # Under fusion, `vocab_size`/`pad_id` above are the *phone*
        # vocabulary's; the Qwen side never becomes an embedding table in
        # the model (it enters as pooled features), so there is no second
        # vocab_size to thread through.
        tokenizer_config["text_frontend"] = "fusion"
        tokenizer_config["lm_hidden_size"] = lm_extractor.hidden_size
        tokenizer_config["gate_init_eps"] = params.gate_init_eps
    params.update(tokenizer_config)

    logging.info(params)

    logging.info("About to create model")

    model = ZipVoice(
        **model_config["model"],
        **tokenizer_config,
    )

    if params.checkpoint is not None:
        logging.info(f"Loading pre-trained model from {params.checkpoint}")
        _ = load_checkpoint(filename=params.checkpoint, model=model, strict=True)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    if params.start_epoch > 1:
        if params.resume_from_checkpoint is not None:
            checkpoints = resume_checkpoint(
                params=params,
                model=model,
                model_avg=model_avg,
                resume_from_checkpoint=params.resume_from_checkpoint,
            )
        else:
            checkpoints = resume_checkpoint(
                params=params, model=model, model_avg=model_avg
            )

        check_resume_text_frontend(params, checkpoints)

    model = model.to(params.device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    assert params.lr_hours >= 0

    if params.finetune:
        scheduler = FixedLRScheduler(optimizer)
    elif params.lr_hours > 0:
        scheduler = Eden(
            optimizer, params.lr_batches, params.lr_hours, params.warmup_batches
        )
    else:
        scheduler = Eden(
            optimizer, params.lr_batches, params.lr_epochs, params.warmup_batches
        )

    scaler = create_grad_scaler(enabled=params.use_fp16)

    if params.start_epoch > 1 and checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

        if "grad_scaler" in checkpoints:
            logging.info("Loading grad scaler state dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    def remove_short_and_long_utt(c: Cut, min_len: float, max_len: float):
        if c.duration < min_len or c.duration > max_len:
            return False
        return True

    _remove_short_and_long_utt = partial(
        remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len
    )

    datamodule = TtsDataModule(args)
    if params.dataset == "emilia":
        train_cuts = CutSet.mux(
            datamodule.train_emilia_EN_cuts(),
            datamodule.train_emilia_ZH_cuts(),
            weights=[46000, 49000],
        )
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = CutSet.mux(
            datamodule.dev_emilia_EN_cuts(),
            datamodule.dev_emilia_ZH_cuts(),
            weights=[0.5, 0.5],
        )
    elif params.dataset == "libritts":
        train_cuts = datamodule.train_libritts_cuts()
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = datamodule.dev_libritts_cuts()
    else:
        assert params.dataset == "custom"
        train_cuts = datamodule.train_custom_cuts(params.train_manifest)
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = datamodule.dev_custom_cuts(params.dev_manifest)
        # To avoid OOM issues due to too long dev cuts
        dev_cuts = dev_cuts.filter(_remove_short_and_long_utt)

    if params.tokenizer in ["emilia", "espeak", "dialog"]:
        if not hasattr(train_cuts[0].supervisions[0], "tokens") or not hasattr(
            dev_cuts[0].supervisions[0], "tokens"
        ):
            logging.warning(
                f"Using {params.tokenizer} tokenizer but tokens are not prepared,"
                f"will tokenize on-the-fly, which can slow down training significantly."
            )
    lang_rng = None
    if params.tokenizer == "fusion":
        # ADR point 2: the phone branch always phonemizes under the immutable
        # `supervision.language`, so a sampled `[LANG:auto]`/wrong tag on the
        # lm_token side would put the two branches on different premises for
        # the same utterance. Enforced in code rather than left to shell
        # convention, because it is silent and load-bearing if violated.
        if params.lang_auto_prob != 0.0 or params.lang_wrong_prob != 0.0:
            raise ValueError(
                "--tokenizer=fusion requires --lang-auto-prob=0 and "
                "--lang-wrong-prob=0 (got "
                f"{params.lang_auto_prob} / {params.lang_wrong_prob}). The "
                "phone branch is always built from the true "
                "supervision.language, so a sampled or deliberately-wrong "
                "language tag would desynchronize the two branches. See the "
                "ADR's point 2, 'Config enforcement required for "
                "training-time consistency'."
            )
    if params.tokenizer == "multilingual":
        assert 0.0 <= params.lang_auto_prob <= 1.0, params.lang_auto_prob
        assert 0.0 <= params.lang_wrong_prob <= 1.0, params.lang_wrong_prob
        assert params.lang_auto_prob + params.lang_wrong_prob <= 1.0, (
            params.lang_auto_prob,
            params.lang_wrong_prob,
        )
        # A single persistent Random instance, not re-seeded per call, so
        # repeated epochs (train/dev cuts are lazy and re-tokenized each
        # time they're iterated) keep advancing rather than repeating the
        # same dropout pattern every epoch.
        lang_rng = random.Random(params.seed)
        logging.info(
            f"[LANG:xx] tags derived per-utterance from each cut's "
            f"supervision.language field (auto_prob={params.lang_auto_prob}, "
            f"wrong_prob={params.lang_wrong_prob}); dev/test cuts always get "
            f"the ground-truth tag (auto_prob=wrong_prob=0) with their own "
            f"RNG, so validation never perturbs the training dropout stream."
        )

    if params.tokenizer == "fusion":
        # No tag sampling here at all: the fusion frontend's `[LANG:xx]` is
        # deterministic (the ADR requires --lang-auto-prob/--lang-wrong-prob
        # to be 0, enforced above) and `FusionTokenizer` prepends it itself.
        _tokenize_text_train = partial(
            tokenize_text_fusion, tokenizers=fusion_tokenizers
        )
        _tokenize_text_dev = _tokenize_text_train
    else:
        _tokenize_text_train = partial(
            tokenize_text,
            tokenizer=tokenizer,
            lang_auto_prob=params.lang_auto_prob,
            lang_wrong_prob=params.lang_wrong_prob,
            lang_rng=lang_rng,
        )
        _tokenize_text_dev = partial(
            tokenize_text,
            tokenizer=tokenizer,
            lang_auto_prob=0.0,
            lang_wrong_prob=0.0,
            lang_rng=random.Random(params.seed) if lang_rng is not None else None,
        )
    train_cuts = train_cuts.map(_tokenize_text_train)
    dev_cuts = dev_cuts.map(_tokenize_text_dev)

    train_dl = datamodule.train_dataloaders(train_cuts)

    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    if params.scan_oom:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
            lm_extractor=lm_extractor,
        )

    logging.info("Training started")

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        if params.lr_hours == 0:
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
            lm_extractor=lm_extractor,
        )

        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
