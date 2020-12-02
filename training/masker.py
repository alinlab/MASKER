import torch
import torch.nn as nn
import torch.nn.functional as F
from training.common import AverageMeter, one_hot, uniform_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_masker(args, loader, model, optimizer, epoch=0):
    model.train()

    if isinstance(model, nn.DataParallel):
        n_classes = model.module.n_classes
    else:
        n_classes = model.n_classes

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['ssl'] = AverageMeter()
    losses['ent'] = AverageMeter()

    for i, (tokens, labels) in enumerate(loader):
        batch_size = tokens.size(0)
        tokens = tokens.to(device)
        labels = labels.to(device)

        labels_ssl = labels[:, :-1]  # self-sup labels (B, K)
        labels_cls = labels[:, -1]  # class labels (B)

        out_cls, out_ssl, out_ood = model(tokens, training=True)

        # classification loss

        if args.classifier_type == 'softmax':
            loss_cls = F.cross_entropy(out_cls, labels_cls)
        elif args.classifier_type == 'regression':
            out_cls = out_cls.squeeze()
            loss_cls = F.mse_loss(out_cls, labels_cls.float())
        else:
            labels_cls = one_hot(labels_cls, n_classes=n_classes)
            loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels_cls)

        # self-supervision loss
        out_ssl = out_ssl.permute(0, 2, 1)
        loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1)  # ignore non-masks (-1)
        loss_ssl = loss_ssl * args.lambda_ssl

        # outlier regularization loss
        if args.classifier_type!='regression':
            out_ood = F.log_softmax(out_ood, dim=1)  # log-probs
            unif = uniform_labels(labels, n_classes=n_classes)
            loss_ent = F.kl_div(out_ood, unif)
            loss_ent = loss_ent * args.lambda_ent
            loss = loss_cls + loss_ssl + loss_ent
        else:
            loss_ent=torch.FloatTensor([0]).to(device)
            loss = loss_cls + loss_ssl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses['cls'].update(loss_cls.item(), batch_size)
        losses['ssl'].update(loss_ssl.item(), batch_size)
        losses['ent'].update(loss_ent.item(), batch_size)

    print('[Epoch %2d] [LossC %f] [LossS %f] [LossE %f]' %
          (epoch, losses['cls'].average, losses['ssl'].average, losses['ent'].average))

