import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from accelerate import Accelerator
import numpy as np


def mixup_patch_multi_short(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    top_k_patch: torch.Tensor,
    saliency_mixup,
    size: int = 32,
    num_patches: int = 8,
    NUM_CLASS=1000,
):
    """
    Input Mixup a batch on patch level
    One top_k_patch for each image

    x: input batch, shape (B, 3, W, H)
    y: scalar batch labels, shape (B,)
    saliency: saliency of x, shape (B, patch_size, patch_size)
    top_k_patch: vector in range [0.0, 1.0]. Top-k patch values for each image in the batch
    size: int, default: 32. Image size.
    num_patches: int, default: 8. Number of patches to divide the image. The image will be
        divided into `num_patches x num_patches` patches.
    """
    device = "cuda"
    DIRICHLET_ALPHA = 0.2
    batch_size = inputs.size()[0]

    with torch.no_grad():
        one_hot_y = F.one_hot(labels, NUM_CLASS)
        # top_k_patch = torch.tensor(top_k_patch, dtype=torch.float32).to(device)

        # Sample weights
        w1, w2 = torch.distributions.dirichlet.Dirichlet(
            torch.tensor([DIRICHLET_ALPHA, DIRICHLET_ALPHA], device=device).float()
        ).sample()
        saliency_flat = torch.reshape(saliency_mixup, (batch_size, -1))  # Flatten
        quants = torch.quantile(
            saliency_flat, top_k_patch.float(), dim=1, keepdim=True
        )  # Calculate quantile
        quants = quants[
            torch.arange(batch_size), torch.arange(batch_size), 0
        ]  # Construct a matrix and get the diag
        mask = (saliency_mixup >= quants[:, None, None]).type(
            torch.float64
        )  # Construct mask, shape: (B, W, H)
        mask = mask.unsqueeze(1)  # Add channel dimension

        # Upsample mask
        mask = F.interpolate(
            mask.float(), scale_factor=size // num_patches, mode="nearest"
        )

        mask_perm = mask.flip(0)
        inputs_perm = inputs.flip(0)
        one_hot_y_perm = one_hot_y.flip(0)

        mask_inter = torch.where(mask == mask_perm, 1, 0)  # Case 1 and 3
        mask_neq = torch.where(mask != mask_perm, 1, 0)  # Case 2

        inputs_eq = mask_inter * (
            w1 * inputs + w2 * inputs_perm
        )  # Mixup case 1 and 3
        inputs_neq = mask_neq * (
            mask * inputs + mask_perm * inputs_perm
        )  # Ignore case 2

        mixed_inputs = inputs_neq + inputs_eq  # Combine results

        mixup_weight = (
            (mask_inter.flatten(1).sum(1) / (size * size))
            .unsqueeze(1)
            .repeat(1, NUM_CLASS)
        )
        x1_weight = (
            ((mask_neq * mask).flatten(1).sum(1) / (size * size))
            .unsqueeze(1)
            .repeat(1, NUM_CLASS)
        )
        x2_weight = (
            ((mask_neq * mask_perm).flatten(1).sum(1) / (size * size))
            .unsqueeze(1)
            .repeat(1, NUM_CLASS)
        )
        mixed_labels = (
            mixup_weight * (w1 * one_hot_y + w2 * one_hot_y_perm)
            + x1_weight * one_hot_y
            + x2_weight * one_hot_y_perm
        )

    return mixed_inputs, mixed_labels


if __name__ == "__main__":
    from models import imagenet_resnet

    model = imagenet_resnet.resnet50(100)

    size = 32
    inputs = torch.rand(100, 3, 32, 32)
    labels = torch.randint(0, 100, (100,))
    actions = torch.randint(0, 10, (100,))
    top_k_patch = []
    num_patches = 8
    action_space_mapper = {"top_k_patch": np.array(list(range(0, 10))) * 0.11}
    for act in actions:
        top_k_patch.append(action_space_mapper["top_k_patch"][act])
    print(
        mixup_patch_multi_short(
            model, inputs, labels, top_k_patch, size=size, num_patches=num_patches
        )
    )
