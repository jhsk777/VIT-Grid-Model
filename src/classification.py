import torch


def categorical_to_continuous(categorical: torch.Tensor, class_boundaries: torch.Tensor, method="mean"):
    """Converts categorical labels to continuous values based on class boundaries.

    Args:
        categorical (torch.Tensor): Categorical labels, with shape [batch_size, *dims]
        class_boundaries (torch.Tensor): Class boundaries, with shape [n_classes].
        method (str, optional): Method for calculating continuous values.
            Options: 'mean' (default) for midpoint between boundaries.

    Returns:
        torch.Tensor: Continuous values corresponding to categorical labels, with same shape as `categorical`.

    Raises:
        ValueError: If `method` is not 'mean'.
    """
    if method == "mean":
        class_boundaries = class_boundaries.float()
        n_classes = len(class_boundaries) + 1
        continuous = torch.zeros_like(categorical, dtype=torch.float32)

        midpoints = (class_boundaries[:-1] + class_boundaries[1:]) / 2
        mid = (categorical != 0) & (categorical != n_classes - 1)
        continuous[categorical == 0] = class_boundaries[0] / 2  # mean of 0 and first boundary
        continuous[mid] = midpoints[categorical[mid] - 1]  # mean of previous and next boundaries
        continuous[categorical == n_classes - 1] = class_boundaries[-1]  # value of last boundary
    else:
        raise ValueError("Invalid method: {}. Only 'mean' is supported.".format(method))

    return continuous
