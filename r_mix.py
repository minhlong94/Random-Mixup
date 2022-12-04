import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

"""
Sample usage:

device = "cuda"
dataloader = ...
model = ...
NUM_CLASS = 1000

for idx, (data, label) in dataloader:
    data, label = data.to(device), label.to(device)
    label_onehot = F.one_hot(label, NUM_CLASS)
    data_mixed, label_mixed = r_mix(model=model, inputs=data, labels=label_onehot)
    output = model(data_mixed)
    loss = nn.BCELoss().to(device)(nn.Softmax(dim=1)(output), label_onehot)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

"""


def r_mix(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_patches: np.random.choice([2, 4, 8, 16, 32]),
    alpha: float = 0.2,
    action_space: np.ndarray = np.linspace(0.0, 0.99, num=10)
):
    """
    model: torch.nn.Module, the classifier
    inputs: torch.Tensor, the input images of shape (BatchSize, Channels, Height, Width)
    labels: torch.Tensor, the output labels, one-hot encoded. Simply use: label = F.one_hot(label_scalar, NUM_CLASS).float()
    alpha: float, beta's alpha coefficient: Beta(alpha, alpha). Should be 2.0 for CIFAR and 0.2 for others.
    num_patches: number of patches to divide the image into. Note that kernel_size = image_size // num_patches
    action_space: np.ndarray, the space to sample the percentile values
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = inputs.shape[0]
    size = inputs.shape[-1]
    num_class = labels.shape[1]
    top_k_patch = torch.tensor(np.random.choice(action_space, size=inputs.shape[0]), device=device)

    # Generate the Saliency Map
    def compute_saliency(model, inputs, labels):
        CELoss_saliency = nn.CrossEntropyLoss(reduction="none").to(
            device
        )
        model.train().to(device)
        inputs_next, targets_next = inputs.to(device), labels.to(device)

        input_var = Variable(inputs_next, requires_grad=True)
        target_var = Variable(targets_next)
        output_saliency = model(input_var)

        loss_batch_saliency = (
            2 * CELoss_saliency(output_saliency, target_var) / num_class
        )
        loss_batch_mean_saliency = torch.mean(loss_batch_saliency, dim=0)
        model.zero_grad(set_to_none=True)
        loss_batch_mean_saliency.backward()

        # Compute saliency as L2 norm of grad across channels like Co-Mixup
        saliency = torch.sqrt(
            torch.mean(input_var.grad ** 2, dim=1)
        )

        with torch.no_grad():
            # Down-sample
            saliency = F.avg_pool2d(saliency, kernel_size=size // num_patches)

            # Normalize to sum up to 1
            saliency = saliency / saliency.reshape(inputs.size()[0], -1).sum(1).reshape(
                batch_size, 1, 1
            )

        return saliency

    inputs_norm = inputs.to(device)
    labels = labels.to(device)

    saliency_mixup = compute_saliency(model, inputs, labels)
    with torch.no_grad():
        one_hot_y = F.one_hot(labels, num_class)

        # Sample weights
        lam = torch.distributions.beta.Beta(
            torch.as_tensor([alpha], device=device).float(),
            torch.as_tensor([alpha], device=device).float(),
        ).sample()
        w1, w2 = lam, 1 - lam

        # Flatten the map
        saliency_flat = torch.reshape(saliency_mixup, (batch_size, -1))

        # Calculate quantile
        quants = torch.quantile(
            saliency_flat, top_k_patch, dim=1, keepdim=True
        )

        # Construct a matrix and get the diagonal
        quants = quants[
            torch.arange(batch_size), torch.arange(batch_size), 0
        ]

        # Construct mask: 1 if the value is in the top salient regions, 0 otherwise. Shape: (B, W, H)
        mask = (saliency_mixup >= quants[:, None, None]).type(
            torch.float64
        )
        mask = mask.unsqueeze(1)  # Add channel dimension

        # Up-sample mask
        mask = F.interpolate(
            mask.float(), scale_factor=size // num_patches, mode="nearest"
        )

        index = torch.randperm(batch_size).to("cuda")
        mask_perm = mask[index, :]
        inputs_perm = inputs_norm[index, :]
        one_hot_y_perm = one_hot_y[index, :]

        mask_inter = torch.where(mask == mask_perm, 1, 0)  # Case 1 and 3, if both belongs to top and least
        mask_neq = torch.where(mask != mask_perm, 1, 0)  # Case 2, if one belongs to top and one belongs to least

        # Mix case 1 and 3
        inputs_eq = mask_inter * (
            w1 * inputs_norm + w2 * inputs_perm
        )

        # Only choose the patch that belongs to the top salient region
        inputs_neq = mask_neq * (
            mask * inputs_norm + mask_perm * inputs_perm
        )

        # Combine results
        mixed_inputs = inputs_neq + inputs_eq

        # Weight for the labels
        mixup_weight = (
            (mask_inter.flatten(1).sum(1) / (size * size))
            .unsqueeze(1)
            .repeat(1, num_class)
        )
        x1_weight = (
            ((mask_neq * mask).flatten(1).sum(1) / (size * size))
            .unsqueeze(1)
            .repeat(1, num_class)
        )
        x2_weight = (
            ((mask_neq * mask_perm).flatten(1).sum(1) / (size * size))
            .unsqueeze(1)
            .repeat(1, num_class)
        )
        mixed_labels = (
            mixup_weight * (w1 * one_hot_y + w2 * one_hot_y_perm)
            + x1_weight * one_hot_y
            + x2_weight * one_hot_y_perm
        )

    return mixed_inputs, mixed_labels


