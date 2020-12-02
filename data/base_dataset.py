import os
import json
from abc import *

import torch
from torch.utils.data import TensorDataset
import numpy as np

from common import DATA_PATH


def tokenize(tokenizer, raw_text):
    max_len = tokenizer.max_len

    if len(raw_text) > max_len:
        raw_text = raw_text[:max_len]

    tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    tokens = torch.tensor(tokens).long()

    if tokens.size(0) < max_len:
        padding = torch.zeros(max_len - tokens.size(0)).long()
        padding.fill_(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        tokens = torch.cat([tokens, padding])

    return tokens


def create_tensor_dataset(inputs, labels):
    assert len(inputs) == len(labels)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(inputs, labels)

    return dataset


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, data_name, total_class, tokenizer,
                 split_ratio=1.0, seed=0, remain=False, test_only=False):

        self.data_name = data_name
        self.total_class = total_class
        self.root_dir = os.path.join(DATA_PATH, data_name)

        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
        self.seed = seed
        self.remain = remain
        self.test_only = test_only

        self.n_classes = int(self.total_class * self.split_ratio)
        if self.remain is True:
            self.n_classes = self.total_class - self.n_classes

        if self.split_ratio < 1.0:
            self.class_idx = self._get_subclass()
        else:
            self.class_idx = list(range(self.n_classes))

        if not self._check_exists():
            self._preprocess()

        if not self.test_only:
            self.train_dataset = torch.load(self._train_path)
        else:
            self._train_dataset = None

        self.test_dataset = torch.load(self._test_path)

    def _get_subclass(self):
        np.random.seed(self.seed)  # fix random seed
        class_idx = np.random.permutation(self.total_class)

        if self.remain is False:
            class_idx = class_idx[:self.n_classes]  # first selected classes
        else:
            class_idx = class_idx[-self.n_classes:]  # last remaining classes

        return np.sort(class_idx).tolist()

    @property
    def base_path(self):
        if self.split_ratio < 1.0:
            base_path = '{}_{}_sub_{:.2f}_seed_{:d}'.format(
                self.data_name, self.tokenizer.name, self.split_ratio, self.seed)
            if self.remain:
                base_path += '_remain'
        else:
            base_path = '{}_{}'.format(self.data_name, self.tokenizer.name)

        return base_path

    @property
    def _train_path(self):
        return os.path.join(self.root_dir, self.base_path + '_train.pth')

    @property
    def _test_path(self):
        return os.path.join(self.root_dir, self.base_path + '_test.pth')

    def _check_exists(self):
        if not self.test_only and not os.path.exists(self._train_path):
            return False
        elif not os.path.exists(self._test_path):
            return False
        else:
            return True

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def _load_dataset(self, *args, **kwargs):
        pass


class NewsDataset(BaseDataset):
    def __init__(self, tokenizer, split_ratio=1.0, seed=0,
                 test_only=False, remain=False):
        super(NewsDataset, self).__init__('news', 20, tokenizer, split_ratio, seed,
                                          test_only=test_only, remain=remain)

    def _preprocess(self):
        print('Pre-processing news dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, '{}.csv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split(',')

            if not int(toks[1]) in self.class_idx:  # only selected classes
                continue

            path = os.path.join(self.root_dir, '{}'.format(toks[0]))
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            if not raw_text:
                text = tokenize(self.tokenizer, text)

            label = self.class_idx.index(int(toks[1]))  # convert to subclass index
            label = torch.tensor(label).long()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset


class ReviewDataset(BaseDataset):
    def __init__(self, tokenizer, split_ratio=1.0, seed=0,
                 test_only=False, remain=False):
        self.train_test_ratio = 0.7  # split ratio for train/test dataset
        super(ReviewDataset, self).__init__('review', 50, tokenizer, split_ratio, seed,
                                            test_only=test_only, remain=remain)

    def _preprocess(self):
        print('Pre-processing review dataset...')
        source_path = os.path.join(self.root_dir, '50EleReviews.json')
        with open(source_path, encoding='utf-8') as f:
            docs = json.load(f)

        np.random.seed(self.seed)  # fix random seed

        train_inds = []
        test_inds = []

        per_class = 1000  # samples are ordered by class
        for cls in self.class_idx:  # only selected classes
            shuffled = np.random.permutation(per_class)
            num = int(self.train_test_ratio * per_class)

            train_inds += (cls * per_class + shuffled[:num]).tolist()
            test_inds += (cls * per_class + shuffled[num:]).tolist()

        train_dataset = self._load_dataset(docs, train_inds, 'train')
        test_dataset = self._load_dataset(docs, test_inds, 'test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, docs, indices, mode='train', raw_text=False):
        assert mode in ['train', 'test']

        inputs = []
        labels = []

        for i in indices:
            if raw_text:
                text = docs['X'][i]
            else:
                text = tokenize(self.tokenizer, docs['X'][i])

            label = self.class_idx.index(int(docs['y'][i]))  # convert to subclass index
            label = torch.tensor(label).long()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset


class IMDBDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        self.class_dict = {'pos': 1, 'neg': 0}
        super(IMDBDataset, self).__init__('imdb', 2, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing imdb dataset...')
        train_dataset, test_dataset = self._load_dataset('both')
        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='both', raw_text=False):
        assert mode in ['both']

        source_path = os.path.join(self.root_dir, 'imdb.txt')
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        train_inputs = []
        train_labels = []
        test_inputs = []
        test_labels = []

        for line in lines:
            toks = line.split('\t')

            if len(toks) > 5:  # text contains tab
                text = '\t'.join(toks[2:-2])
                toks = toks[:2] + [text] + toks[-2:]

            if raw_text:
                text = toks[2]
            else:
                text = tokenize(self.tokenizer, toks[2])

            if toks[3] == 'unsup':
                continue
            else:
                label = self.class_dict[toks[3]]  # convert to class index
                label = torch.tensor(label).long()

            if toks[1] == 'train':
                train_inputs.append(text)
                train_labels.append(label)
            else:
                test_inputs.append(text)
                test_labels.append(label)

        if raw_text:
            train_dataset = zip(train_inputs, train_labels)
            test_dataset = zip(test_inputs, test_labels)
        else:
            train_dataset = create_tensor_dataset(train_inputs, train_labels)
            test_dataset = create_tensor_dataset(test_inputs, test_labels)

        return train_dataset, test_dataset


class SST2Dataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        super(SST2Dataset, self).__init__('sst2', 2, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing sst2 dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('dev')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'dev']

        source_path = os.path.join(self.root_dir, 'sst2_{}.tsv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split('\t')

            if raw_text:
                text = toks[0]
            else:
                text = tokenize(self.tokenizer, toks[0])

            label = torch.tensor(int(toks[1])).long()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset


class FoodDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        super(FoodDataset, self).__init__('food', 2, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing food dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, 'foods_{}.txt'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split(':')

            if int(toks[1]) == 1:  # pre-defined class 0
                label = 0
            elif int(toks[1]) == 5:  # pre-defined class 1
                label = 1
            else:
                continue

            if raw_text:
                text = toks[0]
            else:
                text = tokenize(self.tokenizer, toks[0])

            label = torch.tensor(label).long()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset


class ReutersDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=True):
        assert test_only is True  # no train dataset
        super(ReutersDataset, self).__init__('reuters', 2, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing reuters dataset...')
        test_dataset = self._load_dataset('test')
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='test', raw_text=False):
        assert mode in ['test']

        inputs = []
        labels = []

        base_path = os.path.join(self.root_dir, 'reuters_test')
        for fname in os.listdir(base_path):
            path = os.path.join(base_path, fname)
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            if not raw_text:
                text = tokenize(self.tokenizer, text)

            label = torch.tensor(-1).float()  # OOD class: -1

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset

class MSRvidDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        super(MSRvidDataset, self).__init__('msrvid', 1, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing STS-B MSRvid dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False): #filename should be in ['Images, MSRvid, Headlines, MSRpar']
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, '{}.tsv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split('\t')

            if not toks[2] == 'MSRvid':
                continue

            try:
                label = float(toks[-1])
            except ValueError:
                continue

            if raw_text:
                text = toks[7]+toks[8]
            else:
                text = toks[7]+self.tokenizer.sep_token+toks[8]
                text = tokenize(self.tokenizer, text)

            label = torch.tensor(label).float()

            inputs.append(text)
            labels.append(label)
        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset

class ImagesDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        super(ImagesDataset, self).__init__('images', 1, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing STS-B Images dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False): #filename should be in ['Images, MSRvid, Headlines, MSRpar']
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, '{}.tsv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split('\t')

            if not toks[2] == 'images':
                continue

            try:
                label = float(toks[-1])
            except ValueError:
                continue

            if raw_text:
                text = toks[7]+toks[8]
            else:
                text = toks[7]+self.tokenizer.sep_token+toks[8]
                text = tokenize(self.tokenizer, text)

            label = torch.tensor(label).float()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset

class MSRparDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        super(MSRparDataset, self).__init__('msrpar', 1, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing STS-B MSRpar dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False): #filename should be in ['Images, MSRvid, Headlines, MSRpar']
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, '{}.tsv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split('\t')

            if not toks[2] == 'MSRpar':
                continue

            try:
                label = float(toks[-1])
            except ValueError:
                continue

            if raw_text:
                text = toks[7]+toks[8]
            else:
                text = toks[7]+self.tokenizer.sep_token+toks[8]
                text = tokenize(self.tokenizer, text)

            label = torch.tensor(label).float()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset

class HeadlinesDataset(BaseDataset):
    def __init__(self, tokenizer, test_only=False):
        super(HeadlinesDataset, self).__init__('headlines', 1, tokenizer, test_only=test_only)

    def _preprocess(self):
        print('Pre-processing STS-B Headlines dataset...')
        train_dataset = self._load_dataset('train')
        test_dataset = self._load_dataset('test')

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False): #filename should be in ['Images, MSRvid, Headlines, MSRpar']
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, '{}.tsv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        inputs = []
        labels = []

        for line in lines:
            toks = line.split('\t')

            if not toks[2] == 'headlines':
                continue

            try:
                label = float(toks[-1])
            except ValueError:
                continue

            if raw_text:
                text = toks[7]+toks[8]
            else:
                text = toks[7]+self.tokenizer.sep_token+toks[8]
                text = tokenize(self.tokenizer, text)

            label = torch.tensor(label).float()

            inputs.append(text)
            labels.append(label)

        if raw_text:
            dataset = zip(inputs, labels)
        else:
            dataset = create_tensor_dataset(inputs, labels)

        return dataset