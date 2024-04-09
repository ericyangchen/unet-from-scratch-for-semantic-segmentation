import numpy as np
import torch


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dice_score(pred_mask, gt_mask):
    """
    Args:
        pred_mask: torch.Tensor with shape (N, C, H, W), Channel dimension is 1
        gt_mask: torch.Tensor with shape (N, C, H, W), Channel dimension is 1
    """
    epsilon = 1e-8

    sum_dim = (-1, -2, -3)

    intersection = (pred_mask * gt_mask).sum(dim=sum_dim)
    union = pred_mask.sum(dim=sum_dim) + gt_mask.sum(dim=sum_dim)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    # output: shape (N,) - dice score for each image in the batch (N images)
    return dice


def convert_output_to_binary_mask(outputs, threshold=0.5):
    binary_masks = torch.nn.functional.sigmoid(outputs) > threshold

    return binary_masks
