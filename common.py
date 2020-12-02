import argparse

DATA_PATH = './dataset'
CKPT_PATH = './checkpoint'


def parse_args(mode):
    assert mode in ['train', 'eval']

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help='dataset (news|review|imdb|etc.)',
                        required=True, type=str)
    parser.add_argument("--split_ratio", help='split ratio for ID/OOD sets',
                        default=1.0, type=float)
    parser.add_argument("--backbone", help='backbone network',
                        choices=['bert', 'roberta', 'albert'],
                        default='bert', type=str)
    parser.add_argument("--classifier_type", help='classifier type (softmax|sigmoid)',
                        choices=['softmax', 'sigmoid', 'regression'],
                        default='sigmoid', type=str)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    if mode == 'train':
        parser = _parse_args_train(parser)
    else:
        parser = _parse_args_eval(parser)

    return parser.parse_args()


def _parse_args_train(parser):
    parser.add_argument("--train_type", help='train type (base|residual|masker)',
                        choices=['base', 'residual', 'masker'],
                        default='masker', type=str)
    parser.add_argument("--use_biased_dataset", help='use biased dataset to train a biased model',
                        action='store_true')
    parser.add_argument("--optimizer", help='optimizer type (adam_ood|adam_gen)',
                        choices=['adam_vanilla', 'adam_masker'],
                        default='adam_vanilla', type=str)
    parser.add_argument("--epochs", help='training epochs',
                        default=10, type=int)

    parser.add_argument("--keyword_type", help='keyword type (random|tfidf|attention|etc.)',
                        choices=['random', 'tfidf', 'attention'],
                        default='attention', type=str)
    parser.add_argument("--keyword_per_class", help='number of keywords for each class',
                        default=10, type=int)

    parser.add_argument("--biased_model_path", help='path for the pre-trained biased model',
                        default=None, type=str)

    parser.add_argument("--attn_backbone", help='backbone for attention network (None: args.backbone)',
                        default=None, type=str)
    parser.add_argument("--attn_model_path", help='path for the pre-trained attention model',
                        default=None, type=str)
    parser.add_argument("--lambda_ssl", help='weight for keyword reconstruction loss',
                        default=0.001, type=float)
    parser.add_argument("--lambda_ent", help='weight for entropy regularization loss',
                        default=0.001, type=float)
    return parser


def _parse_args_eval(parser):
    parser.add_argument("--eval_type", help='evaluation type (acc|ood)',
                        choices=['acc', 'ood', 'regression'],
                        default='acc', type=str)
    parser.add_argument("--model_path", help='path for the pre-trained model',
                        default=None, type=str)
    parser.add_argument("--test_dataset", help='dataset for classification',
                        default=None, type=str)
    parser.add_argument("--ood_datasets", help='datasets for OOD detection',
                        default=None, nargs="*", type=str)
    return parser

