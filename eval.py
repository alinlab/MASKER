import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_base_dataset
from models import load_backbone, BaseNet
from evals import test_acc, compute_aurocs, test_pearson

from common import CKPT_PATH, parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args(mode='eval')

    args.batch_size = 16

    print('Loading dataset and model...')
    backbone, tokenizer = load_backbone(args.backbone)
    dataset = get_base_dataset(args.dataset, tokenizer, args.split_ratio, args.seed, test_only=True)
    model = BaseNet(args.backbone, backbone, dataset.n_classes).to(device)

    assert args.model_path is not None
    state_dict = torch.load(os.path.join(CKPT_PATH, args.dataset, args.model_path))

    for key in list(state_dict.keys()):  # only keep base parameters
        if key.split('.')[0] not in ['backbone', 'dense', 'net_cls']:
            state_dict.pop(key)

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    test_loader = DataLoader(dataset.test_dataset, shuffle=False,
                             batch_size=args.batch_size, num_workers=4)

    print('Evaluate {}...'.format(args.eval_type))
    if args.eval_type == 'acc':
        dataset = get_base_dataset(args.test_dataset, tokenizer, args.split_ratio, args.seed, test_only=True)
        test_loader = DataLoader(dataset.test_dataset, shuffle=False,
                             batch_size=args.batch_size, num_workers=4)
        acc = test_acc(test_loader, model)
        print('test acc: {:.2f}'.format(acc))

    elif args.eval_type == 'ood':
        ood_loaders = dict()
        for ood_name in args.ood_datasets:
            if ood_name == 'remain':
                ood_dataset = get_base_dataset(args.dataset, tokenizer, args.split_ratio, args.seed,
                                               test_only=True, remain=True)  # remaining of ID dataset
            else:
                ood_dataset = get_base_dataset(ood_name, tokenizer, args.split_ratio, args.seed,
                                               test_only=True)  # OOD dataset
            ood_loader = DataLoader(ood_dataset.test_dataset, shuffle=False,
                                    batch_size=args.batch_size, num_workers=4)
            ood_loaders[ood_name] = ood_loader

        aurocs = compute_aurocs(model, test_loader, ood_loaders, args.classifier_type)

        for ood_name, auroc in aurocs.items():
            print('auroc ({}): {:.2f}'.format(ood_name, auroc))

    elif args.eval_type == 'regression':
        dataset = get_base_dataset(args.test_dataset, tokenizer, args.split_ratio, args.seed, test_only=True)
        test_loader = DataLoader(dataset.test_dataset, shuffle=False,
                             batch_size=args.batch_size, num_workers=4)
        corr = test_pearson(test_loader, model)
        print('test corr: {:.4f}'.format(corr))

    else:
        raise ValueError('No matching eval type')


if __name__ == "__main__":
    main()

