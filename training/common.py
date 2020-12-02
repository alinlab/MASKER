import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def one_hot(labels, n_classes):
    one_hot = torch.zeros(labels.size(0), n_classes).to(device)
    one_hot[torch.arange(labels.size(0)), labels] = 1
    return one_hot


def uniform_labels(labels, n_classes):
    unif = torch.ones(labels.size(0), n_classes).to(device)
    return unif / n_classes

