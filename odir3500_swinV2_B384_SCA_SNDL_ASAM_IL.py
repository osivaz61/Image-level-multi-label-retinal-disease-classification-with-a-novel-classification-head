import os
import cv2
import yaml
import math
import time
import torch
import random as rdn
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
from contextlib import suppress
from collections import OrderedDict
from sklearn import model_selection
import torch.utils.data as torchdata
from torch.optim.optimizer import Optimizer
from albumentations.pytorch import ToTensorV2
from timm.utils import AverageMeter, CheckpointSaver, NativeScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import myCheckpointSaver as ckptSaver

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.data import create_loader
from timm.models.layers import convert_sync_batchnorm
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument('--pin-mem', action='store_true', default=True,help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def gather_predict_label(eval_metrics):

    world_size = dist.get_world_size()
    predictions_list = [torch.zeros_like(eval_metrics['predictions']) for _ in range(world_size)]
    validlabel_list = [torch.zeros_like(eval_metrics['valid_label']) for _ in range(world_size)]
    
    dist.all_gather(predictions_list, eval_metrics['predictions'])
    dist.all_gather(validlabel_list, eval_metrics['valid_label'])
    
    return torch.cat(predictions_list),torch.cat(validlabel_list)

class CFG:
    ######################
    # Globals #
    ######################
    seed = 1213
    epochs = 40
    train = True
    oof = True
    inference = True
    folds = [0, 1, 2]
    img_size = 384
    testCrop = False
    main_metric = "epoch_score"
    minimize_metric = False
    distribute = False
    rank = -1
    world_size = -1

    ######################
    # Data #
    ######################
    basePath = "/arf/home/osivaz/data/ODIR/"
    train_datadir = Path(basePath + "/Training Set/Images/")
    test_datadir_off = Path(basePath + "/Off-site Test Set/Images/")
    test_datadir_on = Path(basePath + "/On-site Test Set/Images/")

    ######################
    # Dataset #
    ######################
    target_columns = ["N", "D",	"G", "C", "A", "H", "M", "O"]

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 4,
            "num_workers": 4,
            "is_training": True
        },
        "valid": {
            "batch_size": 16,
            "num_workers": 4,
            "is_training": False
        },
        "test": {
            "batch_size": 8,
            "num_workers": 4,
            "is_training": False
        }
    }

    ######################
    # Split #
    ######################
    split = "MultilabelStratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 1213
    }

    ######################
    # Model #
    ######################
    base_model_name = "tf_efficientnet_b5_ns"
    pooling = "GeM"
    pretrained = True
    num_classes = 8

    ######################
    # Criterion #
    ######################
    loss_name = "BCEFocalLoss"
    loss_params: dict = {}

    ######################
    # Optimizer #
    ######################
    optimizer_name = "Adam"
    base_optimizer = "Adam"
    optimizer_params = {
        "lr": 0.0005
    }
    # For SAM optimizer
    base_optimizer = "Adam"

    ######################
    # Scheduler #
    ######################
    scheduler_name = "CosineAnnealingLR"
    scheduler_params = {
        "T_max": 10
    }


def crop_image_from_gray(image: np.ndarray, threshold: int = 7):
    if image.ndim == 2:
        mask = image > threshold
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image > threshold

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape == 0):
        return image
    else:
        image1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        image3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

        image = np.stack([image1, image2, image3], axis=-1)
        return image

from PIL import Image
class TrainDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, target_columns: list, transform = None, center_crop = True):
        self.df = df
        self.filenames = df["ID"].values
        self.datadir = datadir
        self.target_columns = target_columns
        self.labels = df[target_columns].values
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):

        filename = self.filenames[index].split('_')[0]
        pathImage = self.datadir / self.filenames[index]

        imageLeftt = cv2.cvtColor(cv2.imread(str(pathImage)), cv2.COLOR_BGR2RGB)
        if self.center_crop:
            imageLeftt = crop_image_from_gray(imageLeftt)

        pilLeftt = Image.fromarray(imageLeftt)

        if self.transform:
            imageLeftt = self.transform(pilLeftt)

        label = torch.tensor(self.labels[index]).float()
        return {
            "imageLeftt": imageLeftt,
            "targets": label,
            "indxs" : torch.tensor(index)
        }

class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transform=None, center_crop=True):
        self.df = df
        self.filenames = df["ID"].values
        self.datadir = datadir
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        filenameLeftt = str(self.filenames[index]) + "_left"
        filenameRight = str(self.filenames[index]) + "_right"
        pathLeftt = self.datadir / f"{filenameLeftt}.jpg"
        pathRight = self.datadir / f"{filenameRight}.jpg"
        imageLeftt = cv2.cvtColor(cv2.imread(str(pathLeftt)), cv2.COLOR_BGR2RGB)
        imageRight = cv2.cvtColor(cv2.imread(str(pathRight)), cv2.COLOR_BGR2RGB)
        if self.center_crop:
            imageLeftt = crop_image_from_gray(imageLeftt)
            imageRight = crop_image_from_gray(imageRight)

        if self.transform:
            augmented = self.transform(image = imageLeftt)
            imageLeftt = augmented["image"]

            augmented = self.transform(image = imageRight)
            imageRight = augmented["image"]
        return {
            "imageLeftt": imageLeftt,
            "imageRight": imageRight
        }

try:
    from torchvision.transforms import InterpolationMode
    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

# =================================================
# Transforms #
# =================================================
from timm.data import create_transform
from torchvision import transforms
def get_transforms(img_size: int, mode="train"):
    resize_im = CFG.img_size > 32
    if mode == "train":
        transform = create_transform(
            input_size = CFG.img_size,
            is_training=True,
            color_jitter = 0.4,
            auto_augment = 'rand-m9-mstd0.5-inc1',
            re_prob = 0.25,
            re_mode = 'pixel',
            re_count = 1,
            interpolation = 'bicubic',
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(CFG.img_size, padding = 4)
        return transform
    elif mode == "valid":
        t = []
        if resize_im:
            if CFG.testCrop:
                size = int((256 / 224) * CFG.img_size)
                t.append(transforms.Resize(size, interpolation = _pil_interp('bicubic')))
                t.append(transforms.CenterCrop(CFG.img_size))
            else:
                t.append(transforms.Resize((CFG.img_size, CFG.img_size), interpolation = _pil_interp('bicubic')))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        return transforms.Compose(t)
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2()
        ])

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)

from functools import partial
from SSAtMLD import Block_Cls
from timm.models.helpers import load_checkpoint
from swin_transformer_v2_Temp import SwinTransformerV2_Temp
class TimmModel(nn.Module):
    def __init__(self, base_model_name="tf_efficientnet_b0_ns", pooling="GeM", pretrained=True, num_classes=24):
        super().__init__()
        self.swinV2 = SwinTransformerV2_Temp(img_size = CFG.img_size, patch_size = 4, embed_dim = 128, depths = [2, 2, 18, 2],
        window_size = 24, num_heads = [4, 8, 16, 32], mlp_ratio = 4., drop_path_rate = 0.2, pretrained_window_sizes = [12, 12, 12, 6])
        load_checkpoint(self.swinV2, "../preweights/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth", strict = True)
        inFeatures = self.swinV2.head.in_features
        self.swinV2.head = nn.Identity()
        self.classifier = Block_Cls(dim = inFeatures, num_heads = 4, mlp_ratio = 2, qkv_bias = True, qk_scale = True, 
                                    drop = 0.05, attn_drop = 0.0, drop_path = 0.05, norm_layer = partial(nn.LayerNorm, eps=1e-6), sr_ratio = 4).cuda()
        self.query_embed = nn.Embedding(CFG.num_classes, inFeatures)
        self.query_embed.requires_grad_(False)
        self.group_fc = GroupFC(CFG.num_classes)
        self.duplicate_pooling = torch.nn.Parameter(torch.Tensor(CFG.num_classes, inFeatures, 1))
        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(CFG.num_classes))
        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)

    def forward(self, input):
        output = self.swinV2(input)
        tgt = self.query_embed.weight.unsqueeze(0).expand(input.shape[0], -1, -1)
        r = self.classifier(tgt.cuda(), output, 12, 12)
        out_extrap = torch.zeros(r.shape[0], r.shape[1], 1, device = r.device, dtype = r.dtype)
        self.group_fc(r, self.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :CFG.num_classes]
        h_out += self.duplicate_pooling_bias

        return torch.nn.functional.normalize(output.mean(1)), h_out

# =================================================
# Criterion #
# =================================================
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas)**self.gamma * bce_loss + (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss,
}


def get_criterion():
    if hasattr(nn, CFG.loss_name):
        return nn.__getattribute__(CFG.loss_name)(**CFG.loss_params)
    elif __CRITERIONS__.get(CFG.loss_name) is not None:
        return __CRITERIONS__[CFG.loss_name](**CFG.loss_params)
    else:
        raise NotImplementedError


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


version_higher = (torch.__version__ >= "1.5.0")

class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMS enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']
                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)
                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(1 - beta2, grad_residual, grad_residual)
                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:  # Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (
                            1.0 - beta2 ** state['step'])
                    if state['rho_t'] > 4:  # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
                        rt = math.sqrt(rt)
                        step_size = rt * group['lr'] / bias_correction1
                        p.data.addcdiv_(-step_size, exp_avg, denom)
                    else:  # perform SGD style update
                        p.data.add_(-group['lr'], exp_avg)
        return loss




__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief,
    "SAM": SAM,
}


def get_optimizer(model: nn.Module, backbone, classifier):
    optimizer_name = CFG.optimizer_name
    if optimizer_name == "SAM":
        base_optimizer_name = CFG.base_optimizer
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **CFG.optimizer_params)

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **CFG.optimizer_params)
    else:
        return optim.__getattribute__(optimizer_name)([
                    {'params': backbone, 'lr': CFG.optimizer_params['lr']},
                    {'params': classifier, 'lr': CFG.optimizer_params['lr']}
        ])


def get_scheduler(optimizer):
    scheduler_name = CFG.scheduler_name

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **CFG.scheduler_params)


# =================================================
# Split #
# =================================================
def get_split():
    if hasattr(model_selection, CFG.split):
        return model_selection.__getattribute__(CFG.split)(**CFG.scheduler_params)
    else:
        return MultilabelStratifiedKFold(**CFG.split_params)


# =================================================
# Utilities #
# =================================================
def set_seed(seed=42):
    rdn.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

from timm.utils import *
def train_epoch(epoch, model, lemniscate, loader, minimizer, loss_fn, ncaLoss, loss_scaler, lr_scheduler=None, amp_autocast = suppress):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    second_order = hasattr(minimizer.optimizer, 'is_second_order') and minimizer.optimizer.is_second_order

    end = time.time()
    last_idx = len(loader) - 1
    for batch in enumerate(loader):

        batch_idx = batch[0]
        inputLeftt = batch[1]['imageLeftt']
        target = batch[1]['targets']
        index = batch[1]['indxs']
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        inputLeftt, target = inputLeftt.cuda(), target.cuda()
        target = target.float()

        ## with amp_autocast():
        feature, logits = model(inputLeftt)
        output = lemniscate(feature, index)
        loss1 = loss_fn(logits, target)
        loss2 = ncaLoss(output, index)
        loss = loss1 + 0.3 * loss2
        loss.backward()
        minimizer.ascent_step()

        output, logits = model(inputLeftt)
        output = lemniscate(feature, index)
        loss1 = loss_fn(logits, target)
        loss2 = ncaLoss(output, index)
        loss = loss1 + 0.3 * loss2
        loss.backward()
        minimizer.descent_step()
        
        torch.cuda.synchronize()

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % 48 == 0:
            lrl = [param_group['lr'] for param_group in minimizer.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            reduced_loss = reduce_tensor(loss.data, CFG.world_size)
            losses_m.update(reduced_loss.item(), inputLeftt.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=inputLeftt.size(0) / batch_time_m.val,
                        rate_avg=inputLeftt.size(0) / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))
        scheduler.step()
        end = time.time()
    
    if hasattr(minimizer.optimizer, 'sync_lookahead'):
        minimizer.optimizer.sync_lookahead()
    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()    
    preds = []  
    valid_label_ = []
    end = time.time()
    last_idx = len(loader) - 1
    model.eval()

    with torch.no_grad():
        for batch in enumerate(loader):

            batch_idx = batch[0]
            inputLeftt = batch[1]['imageLeftt']
            target = batch[1]['targets']

            last_batch = batch_idx == last_idx
            inputLeftt = inputLeftt.cuda()
            target = target.cuda()
            target = target.float()

            ## with amp_autocast():
            feature, output = model(inputLeftt)

            loss = loss_fn(output, target)
            reduced_loss = reduce_tensor(loss.data, CFG.world_size)
            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputLeftt.size(0))
            preds.append(output.sigmoid().to('cpu').numpy())
            valid_label_.append(target.to('cpu').numpy())

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (args.local_rank == 0 and last_batch or batch_idx % 48 == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m))

    predictions = np.concatenate(preds)
    predictions = torch.from_numpy(predictions)
    predictions = predictions.cuda()

    valid_label_ =  np.concatenate(valid_label_)
    valid_label_ = torch.from_numpy(valid_label_)
    valid_label_ = valid_label_.cuda()

    metrics = OrderedDict([('loss', losses_m.avg), ('predictions', predictions),('valid_label',valid_label_)])
    return metrics


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


from sklearn.metrics import roc_auc_score, average_precision_score
def get_score(y_true, y_pred):
    scores = []
    mAPs = []
    for i in range(y_true.shape[1]):
        try: 
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            mAP = average_precision_score(y_true[:,i], y_pred[:,i], average="macro")
            mAPs.append(mAP)
            scores.append(score)
        except ValueError: 
            pass 
    return np.mean(scores), np.mean(mAPs)

from sklearn.metrics import f1_score
def json_map(cls_id, pred_json, ann_json):
    assert len(ann_json) == len(pred_json)
    predict = pred_json[:, cls_id]
    target = ann_json[:, cls_id]

    tmp = np.argsort(-predict)
    target = target[tmp]
    predict = predict[tmp]

    pre, obj = 0, 0
    for i in range(len(ann_json)):
        if target[i] == 1:
            obj += 1.0
            pre += obj / (i+1)
    pre /= obj
    return pre

def getSegmAP(label, predicted):
    apList = np.zeros(label.shape[1])
    for i in range(label.shape[1]):
        apList[i] = json_map(i, predicted, label)
    return np.mean(apList)

from torch.optim import lr_scheduler
from sklearn import metrics
def odir_metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0
    return kappa, f1, auc, final_score

eps = 1e-8
import math
class NCA_ML_CrossEntropy(nn.Module):
    ''' \sum log(w_{ij}*p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''

    def __init__(self, multiHotMtx, margin=0):
        super().__init__()

        self.register_buffer('multiHotMtx', multiHotMtx)
        # transfer 0,1 multihot to -1,1
        multiHotMtx[multiHotMtx==0] = -1
        self.multiHotMtx = multiHotMtx.cuda()
        self.labelNum = self.multiHotMtx.size(1)
        self.margin = margin

    def forward(self, x, indexes):
        
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        indexes = indexes.cuda()
        batch_multiHotMtx = torch.index_select(self.multiHotMtx.cuda(), 0, indexes.data)

        out = torch.mm(batch_multiHotMtx, self.multiHotMtx.t())
        hamming_dist = (out + self.labelNum) / 2
        weights = hamming_dist / self.labelNum
        
        # print(weights.max(), weights.min())


        # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, weights.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)

        return - loss / batchSize

from torch.autograd import Function
class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N
        
        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        memory = memory.cuda()
        y = y.cuda() 
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

def adjust_memory_update_rate(lemniscate, epoch):
    if epoch >= 40:
        lemniscate.params[1] = 0.7
    if epoch >= 40:
        lemniscate.params[1] = 0.7


import torch
from collections import defaultdict
class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


if __name__ == "__main__":

    _logger = logging.getLogger('train')
    args, args_text = _parse_args()
    # environment
    set_seed(CFG.seed)
    device = get_device()

    # validation
    splitter = get_split()

    alldata = pd.read_csv('newLabel3500.csv')
    comLabels = pd.DataFrame()
    comLabels['ID'] = alldata['ID'].values
    comLabels[CFG.target_columns] = alldata[CFG.target_columns].values

    setup_default_logging()
    _logger.info('osivaz')

    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()

    if not os.path.isdir('ckpt_odir3500_swinV2_B384_SCA_SNDL_ASAM_IL/'):
        if args.local_rank == 0:
            os.mkdir('ckpt_odir3500_swinV2_B384_SCA_SNDL_ASAM_IL/')

    if 'WORLD_SIZE' in os.environ:
        CFG.distribute = int(os.environ['WORLD_SIZE']) > 1
        print(" Distribute  = ", CFG.distribute)

    for cv, (trn_idx, val_idx) in enumerate(splitter.split(comLabels, y = comLabels[CFG.target_columns])):

        device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        if cv == 0:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        CFG.world_size = torch.distributed.get_world_size()
        CFG.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (CFG.rank, CFG.world_size))
        torch.manual_seed(CFG.seed + CFG.rank)

        pathFold = 'ckpt_odir3500_swinV2_B384_SCA_SNDL_ASAM_IL/CVSet' + str(cv)
        if args.local_rank == 0:
            os.mkdir(pathFold)

        model = TimmModel(base_model_name=CFG.base_model_name, pooling=CFG.pooling, pretrained=CFG.pretrained, num_classes=CFG.num_classes)
        if args.local_rank == 0:
            print("Model params = " + str(sum([m.numel() for m in model.parameters()])))
        
        model.cuda()
        model = convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters = True)  # can use device str in Torch >= 1.1

        trn_df = comLabels.loc[trn_idx, :].reset_index(drop = True)
        val_df = comLabels.loc[val_idx, :].reset_index(drop = True)
        dataset_train = TrainDataset(trn_df, CFG.train_datadir, CFG.target_columns, transform = get_transforms(CFG.img_size, "train"))
        dataset_valid = TrainDataset(val_df, CFG.train_datadir, CFG.target_columns, transform = get_transforms(CFG.img_size, "valid"))
        ## trainLoader = torchdata.DataLoader(dataset_train, **CFG.loader_params['train'])
        ## validLoader = torchdata.DataLoader(dataset_valid, **CFG.loader_params['valid'])
        trainLoader = create_loader(dataset_train, input_size = CFG.img_size, transform = get_transforms(CFG.img_size, "train"), **CFG.loader_params['train'], distributed = True, 
                                    use_prefetcher = False, pin_memory = args.pin_mem)
        validLoader = create_loader(dataset_valid, input_size = CFG.img_size, transform = get_transforms(CFG.img_size, "valid"), **CFG.loader_params['valid'], distributed = True, 
                                    use_prefetcher = False, pin_memory = args.pin_mem)
        
        lemniscate = LinearAverage(1024, len(trn_idx), 0.3, 0.4).cuda() ##86.48(0.3 ve 0.4)
        lemniscate = NativeDDP(lemniscate, device_ids = [args.local_rank], find_unused_parameters = True)

        backbone, classifier = [], []
        for name, param in model.named_parameters():
            if 'classifier' in name or 'group_fc' in name or 'linearFCC' in name or 'duplicate_pooling' in name:
                classifier.append(param)
            else:
                backbone.append(param)

        ## criterion = get_criterion()
        criterion = nn.BCEWithLogitsLoss().cuda()
        ncaLoss = NCA_ML_CrossEntropy(torch.tensor(trn_df[CFG.target_columns].values), 0.1 / 0.05).cuda()
        optimizer = get_optimizer(model, backbone, classifier)
        minimizer = ASAM(optimizer, model, rho = 0.5, eta = 0)
        scheduler = lr_scheduler.OneCycleLR(minimizer.optimizer, max_lr = [0.00006, 0.00006], steps_per_epoch = len(trainLoader), epochs = CFG.epochs, pct_start = 0.2)

        eval_metric = 'EVAL_METRIC'
        decreasing = True if eval_metric == 'loss' else False
        saver = ckptSaver.CheckpointSaver(
            model = model, optimizer=optimizer, args = None, model_ema = None, amp_scaler=loss_scaler,
            checkpoint_dir=pathFold, recovery_dir=pathFold, decreasing=decreasing, max_history = 3)

        best_score = 0
        lastBestInd = -1
        for epoch in range(CFG.epochs):
            trainLoader.sampler.set_epoch(epoch)

            adjust_memory_update_rate(lemniscate, epoch)

            train_metrics = train_epoch(epoch + 1, model, lemniscate, trainLoader, minimizer, criterion, ncaLoss, loss_scaler,
                lr_scheduler = scheduler, amp_autocast = amp_autocast)

            eval_metrics = validate(model, validLoader, criterion, amp_autocast=amp_autocast)
            predictions, valid_label = gather_predict_label(eval_metrics)
            valid_label = valid_label.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            AUC, mAP = get_score(valid_label, predictions)
            print("AUC score = %.4f" % AUC)
            print("mAP score = %.4f" % mAP)
            kappa, f1, auc, final_score = odir_metrics(valid_label, predictions)
            print("Kappa score:", kappa)
            print("F-1 score:", f1)
            print("AUC value:", auc)
            print("Final Score:", final_score)

            ## scheduler.step()
            if saver is not None:
                oldBestInd = lastBestInd
                lastBestInd = epoch
                best_score = final_score
                best_metric, best_epoch = saver.save_checkpoint(epoch + 1, metric = best_score)

        del model
        del optimizer
        torch.cuda.empty_cache()
