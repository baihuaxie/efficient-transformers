"""
    Metrics for training

    - loss_fn
    - accuracy

"""

import numpy as np

import torch
import torch.nn as nn

def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss given the predicted log probabilites and labels

    Args:
        outputs: (tensor) dimension = batch_size x 10 -> each element in batch_size dim is a list containing
                  the predicted log probabilities for 10 classes for the image
        labels: (torch.Tensor) dimension = batch_size x 1 -> each element is an integer representing the correct label

    Returns:
        loss: (nn.CrossEntroyLoss) dimension = 1 (scalar) -> a tensor containing the cross entropy loss over the batch

    Note:
        - returned loss must be a class object with a backward() method to facilitate backpropagation
        - can be a torch.tensor with requires_grad=True

    """

    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes the classification accuracy metric

    Args:
        outputs: (tensor) dimension = batch_size x 10 -> each element in batch_size dim is a list containing the predicted log probabilities for 10 classes for the image
        labels: (tensor) dimen = batch_size x 1 -> each element is an integer representing the correct label

    Returns:
        accuracy: (float) accuracy in [0,1]

    """

    _, predicted_labels = torch.max(outputs, dim=1)
    return np.sum(predicted_labels.numpy() == labels.numpy()) / float(labels.numpy().size)


# maitain all metrics used in the training and evaluation loops in this dictionary
metrics = {
    'accuracy': accuracy,
}
