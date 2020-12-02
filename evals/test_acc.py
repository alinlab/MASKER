import torch

from training.common import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_acc(loader, model):
    print('Compute test accuracy...')
    model.eval()

    error_top1 = AverageMeter()

    for i, (tokens, labels) in enumerate(loader):
        batch_size = tokens.size(0)
        tokens = tokens.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(tokens)  # (B, C)

        top1, = acc_k(outputs.data, labels, ks=(1,))

        error_top1.update(top1.item(), batch_size)

    return error_top1.average


def acc_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results
