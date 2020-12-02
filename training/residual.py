import torch
import torch.nn as nn
import torch.nn.functional as F
from training.common import AverageMeter, one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_residual(args, loader, model, biased_model, optimizer, epoch=0):
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

        with torch.no_grad():
            out_b = biased_model(tokens)  # (B, C)
        out_c = model(tokens)  # (B, C)

        # classification loss
        if args.classifier_type == 'softmax':
            p_b = F.softmax(out_b, dim=1)
            p_c = F.softmax(out_c, dim=1)
            out_mult = torch.log(p_b) + torch.log(p_c)  # product of experts
            loss_cls = F.cross_entropy(out_mult, labels)
        else:
            raise NotImplementedError()

        # total loss
        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses['cls'].update(loss_cls.item(), batch_size)

    print('[Epoch %2d] [LossC %f]' %
          (epoch, losses['cls'].average))

