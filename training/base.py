import torch
import torch.nn as nn
import torch.nn.functional as F
from training.common import AverageMeter, one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_base(args, loader, model, optimizer, epoch=0):
    model.train()

    if isinstance(model, nn.DataParallel):
        n_classes = model.module.n_classes
    else:
        n_classes = model.n_classes

    losses = dict()
    losses['cls'] = AverageMeter()

    for i, (tokens, labels) in enumerate(loader):
        batch_size = tokens.size(0)
        tokens = tokens.to(device)
        labels = labels.to(device)

        labels = labels.squeeze(1)  # (B)

        out_cls = model(tokens)  # (B, C)

        # classification loss
        if args.classifier_type == 'softmax':
            loss_cls = F.cross_entropy(out_cls, labels)
        elif args.classifier_type == 'regression':
            out_cls = out_cls.squeeze()
            loss_cls = F.mse_loss(out_cls, labels)
        else:
            labels = one_hot(labels, n_classes=n_classes)
            loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels)

        # total loss
        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses['cls'].update(loss_cls.item(), batch_size)

    print('[Epoch %2d] [LossC %f]' %
          (epoch, losses['cls'].average))

