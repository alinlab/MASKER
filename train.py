  
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get_base_dataset, get_biased_dataset, get_masked_dataset
from models import load_backbone, BaseNet, MaskerNet
from training import train_base, train_residual, train_masker
from evals import test_acc, test_pearson

from common import CKPT_PATH, parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args(mode='train')

    if args.train_type == 'masker':
        if args.keyword_type == 'random':
            args.batch_size = 4
        else:
            args.batch_size = 8
    else:
        args.batch_size = 16

    print('Loading pre-trained backbone network...')
    backbone, tokenizer = load_backbone(args.backbone)

    print('Initializing dataset and model...')
    if args.train_type in ['base', 'residual']:
        # load base/biased dataset and base model
        if not args.use_biased_dataset:
            dataset = get_base_dataset(args.dataset, tokenizer, args.split_ratio, args.seed)
        else:
            dataset = get_biased_dataset(args, args.dataset, tokenizer, args.keyword_type, args.keyword_per_class,
                                         args.split_ratio, args.seed)
        model = BaseNet(args.backbone, backbone, dataset.n_classes).to(device)
        # load biased model
        if args.train_type == 'residual':
            assert args.biased_model_path is not None
            biased_model = BaseNet(args.backbone, backbone, dataset.n_classes).to(device)
            state_dict = torch.load(os.path.join(CKPT_PATH, args.dataset, args.biased_model_path))
            biased_model.load_state_dict(state_dict)
    else:
        # load masked dataset and MASKER model
        dataset = get_masked_dataset(args, args.dataset, tokenizer, args.keyword_type, args.keyword_per_class,
                                     args.split_ratio, args.seed)
        model = MaskerNet(args.backbone, backbone, dataset.n_classes, dataset.n_keywords).to(device)

    if args.optimizer == 'adam_masker':
        optimizer = optim.Adam([
            #{'params': model.parameters()},
            {'params': model.backbone.parameters(), 'lr': 5e-6}
        ], lr=1e-5, eps=1e-8)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-5, eps=1e-8)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if args.train_type == 'residual':
            biased_model = nn.DataParallel(biased_model)

    train_loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                              batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False,
                             batch_size=args.batch_size, num_workers=4)

    print('Training model...')
    for epoch in range(1, args.epochs + 1):
        if args.train_type == 'base':
            train_base(args, train_loader, model, optimizer, epoch)
        elif args.train_type == 'residual':
            train_residual(args, train_loader, model, biased_model, optimizer, epoch)
        else:
            train_masker(args, train_loader, model, optimizer, epoch)

        if args.classifier_type=='regression':
            corr = test_pearson(test_loader, model)
            print('test corr: {:.4f}'.format(corr))
        else:
            acc = test_acc(test_loader, model)
            print('test acc: {:.2f}'.format(acc))

    if isinstance(model, nn.DataParallel):
        model = model.module

    print('Save model...')
    os.makedirs(os.path.join(CKPT_PATH, dataset.data_name), exist_ok=True)

    if args.train_type=='masker':
        model_path = dataset.base_path + '_masker.model'
    elif args.train_type=='base':
        if not args.use_biased_dataset:
            model_path = dataset.base_path + '.model'
        else:
            model_path = dataset.base_path + '_biased.model'
    else:
        model_path = dataset.base_path + '_residual.model'

    save_path = os.path.join(CKPT_PATH, dataset.data_name, model_path)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
