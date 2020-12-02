import os
import random

import torch
from torch.utils.data import TensorDataset


class BiasedDataset(object):
    def __init__(self, base_dataset, keyword):
        assert base_dataset.train_dataset is not None  # train dataset should be exists
        assert keyword.keyword_type != 'random'  # no random keywords

        self.base_dataset = base_dataset
        self.data_name = base_dataset.data_name
        self.base_path = base_dataset.base_path

        self.keyword = keyword.keyword  # keyword values (list)
        self.keyword_type = keyword.keyword_type  # keyword type
        self.n_keywords = len(self.keyword)

        self.tokenizer = base_dataset.tokenizer
        self.n_classes = base_dataset.n_classes

        if not self._check_exists():
            self._preprocess()

        self.train_dataset = torch.load(self._train_path)  # masked dataset
        self.test_dataset = base_dataset.test_dataset

    @property
    def _train_path(self):
        train_path = self.base_dataset._train_path

        keyword_per_class = self.n_keywords // self.n_classes
        suffix = 'biased_{}_{}'.format(self.keyword_type, keyword_per_class)

        train_path = train_path.replace('.pth', '_{}.pth'.format(suffix))
        return train_path

    def _check_exists(self):
        if os.path.exists(self._train_path):
            return True
        else:
            return False

    def _preprocess(self):
        tokenizer = self.base_dataset.tokenizer
        dataset = self.base_dataset.train_dataset

        biased_dataset = _biased_dataset(tokenizer, dataset, keyword=self.keyword)
        torch.save(biased_dataset, self._train_path)


def _biased_dataset(tokenizer, dataset, keyword):

    keyword = dict.fromkeys(keyword, 1)  # convert to dict

    CLS_TOKEN = tokenizer.cls_token_id
    PAD_TOKEN = tokenizer.pad_token_id
    MASK_TOKEN = tokenizer.mask_token_id

    tokens = dataset.tensors[0]
    labels = dataset.tensors[1]

    biased_tokens = []
    biased_labels = []

    for (token, label) in zip(tokens, labels):
        b_token = token.clone()  # biased token (keyword only)

        count = 0  # number of keywords
        for i, tok in enumerate(token):
            if tok == CLS_TOKEN:
                continue
            elif tok == PAD_TOKEN:
                break

            if tok.item() in keyword:
                count += 1
            else:
                b_token[i] = MASK_TOKEN

        if count > 0:  # number of keywords > 0
            biased_tokens.append(b_token)  # (biased)
            biased_labels.append(label)  # (label)

    biased_tokens = torch.stack(biased_tokens)
    biased_labels = torch.stack(biased_labels)

    biased_dataset = TensorDataset(biased_tokens, biased_labels)

    return biased_dataset


