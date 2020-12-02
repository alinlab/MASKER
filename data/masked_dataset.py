import os
import random

import torch
from torch.utils.data import TensorDataset


class MaskedDataset(object):
    def __init__(self, base_dataset, keyword):
        assert base_dataset.train_dataset is not None  # train dataset should be exists

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

        if self.keyword_type == 'random':
            keyword_per_class = self.n_keywords
        else:
            keyword_per_class = self.n_keywords // self.n_classes

        suffix = 'masked_{}_{}'.format(self.keyword_type, keyword_per_class)

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
        seed = self.base_dataset.seed

        if self.keyword_type == 'random':  # mask random words with p = 0.15
            masked_dataset = _masked_dataset(tokenizer, dataset,
                                             seed=seed, key_mask_ratio=0.15)
        else:  # mask keywords with p = 0.5
            masked_dataset = _masked_dataset(tokenizer, dataset, keyword=self.keyword,
                                             seed=seed, key_mask_ratio=0.5)

        torch.save(masked_dataset, self._train_path)


def _masked_dataset(tokenizer, dataset, keyword=None,
                    seed=0, key_mask_ratio=0.5, out_mask_ratio=0.9):

    #keyword_dict = dict.fromkeys(keyword, i for i in range(len(keyword)))  # convert to dict
    keyword_dict={}
    for i,word in enumerate(keyword):
        keyword_dict[word]=i

    keyword=keyword_dict

    CLS_TOKEN = tokenizer.cls_token_id
    PAD_TOKEN = tokenizer.pad_token_id
    MASK_TOKEN = tokenizer.mask_token_id

    random.seed(seed)  # fix random seed
    
    tokens = dataset.tensors[0]
    labels = dataset.tensors[1]

    masked_tokens = []
    masked_labels = []

    for (token, label) in zip(tokens, labels):
        m_token = token.clone()  # masked token (for self-supervision)
        o_token = token.clone()  # outlier token (for entropy regularization)
        m_label = -torch.ones(token.size(0) + 1).long()  # self-sup labels + class label

        for i, tok in enumerate(token):
            if tok == CLS_TOKEN:
                continue
            elif tok == PAD_TOKEN:
                break
            if random.random() < key_mask_ratio:  # randomly mask keywords
                if (keyword is None) or (tok.item() in keyword):  # random MLM or keyword MLM
                    m_token[i] = MASK_TOKEN
                    if keyword is None:
                        m_label[i] = tok  # use full vocabulary
                    else:
                        m_label[i] = keyword[tok.item()] # convert to keyword index

            if (keyword is not None) and (tok.item() not in keyword):
                if random.random() < out_mask_ratio:  # randomly mask non-keywords
                    o_token[i] = MASK_TOKEN

        m_label[-1] = label  # class label

        masked_tokens.append(torch.cat([token, m_token, o_token]))  # (original, masked, outlier)
        masked_labels.append(m_label)

    masked_tokens = torch.stack(masked_tokens)
    masked_labels = torch.stack(masked_labels)

    masked_dataset = TensorDataset(masked_tokens, masked_labels)

    return masked_dataset


