import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.base_dataset import NewsDataset, ReviewDataset, IMDBDataset, SST2Dataset, FoodDataset, ReutersDataset, MSRvidDataset, ImagesDataset, MSRparDataset, HeadlinesDataset
from data.masked_dataset import MaskedDataset
from data.biased_dataset import BiasedDataset
from models import load_backbone

from common import CKPT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_base_dataset(data_name, tokenizer, split_ratio=1.0, seed=0, test_only=False, remain=False):

    print('Initializing base dataset... (name: {})'.format(data_name))

    if data_name == 'news':
        dataset = NewsDataset(tokenizer, split_ratio, seed, test_only=test_only, remain=remain)
    elif data_name == 'review':
        dataset = ReviewDataset(tokenizer, split_ratio, seed, test_only=test_only, remain=remain)
    elif data_name == 'imdb':
        dataset = IMDBDataset(tokenizer, test_only=test_only)
    elif data_name == 'sst2':
        dataset = SST2Dataset(tokenizer, test_only=test_only)
    elif data_name == 'food':
        dataset = FoodDataset(tokenizer, test_only=test_only)
    elif data_name == 'reuters':
        dataset = ReutersDataset(tokenizer, test_only=test_only)
    elif data_name == 'msrvid':
        dataset = MSRvidDataset(tokenizer, test_only=test_only)
    elif data_name == 'images':
        dataset = ImagesDataset(tokenizer, test_only=test_only)
    elif data_name == 'msrpar':
        dataset = MSRparDataset(tokenizer, test_only=test_only)
    elif data_name == 'headlines':
        dataset = HeadlinesDataset(tokenizer, test_only=test_only)
    else:
        raise ValueError('No matching dataset')

    return dataset

def get_biased_dataset(args, data_name, tokenizer, keyword_type, keyword_per_class, split_ratio=1.0, seed=0):
    dataset = get_base_dataset(data_name, tokenizer, split_ratio, seed)  # base dataset

    print('Initializing biased dataset... (name: {})'.format(data_name))
    start_time = time.time()

    keyword = get_keyword(args, dataset, tokenizer, keyword_type, keyword_per_class)
    biased_dataset = BiasedDataset(dataset, keyword)

    print('{:d}s elapsed'.format(int(time.time() - start_time)))
    return biased_dataset

def get_masked_dataset(args, data_name, tokenizer, keyword_type, keyword_per_class, split_ratio=1.0, seed=0):

    dataset = get_base_dataset(data_name, tokenizer, split_ratio, seed)  # base dataset

    print('Initializing masked dataset... (name: {})'.format(data_name))

    if keyword_type == 'random':
        keyword_per_class = len(tokenizer)  # full words

    keyword_path = '{}_{}_{}.pth'.format(data_name, keyword_type, keyword_per_class)
    keyword_path = os.path.join(dataset.root_dir, keyword_path)

    if os.path.exists(keyword_path):
        keyword = torch.load(keyword_path)
    else:
        keyword = get_keyword(args, dataset, tokenizer, keyword_type, keyword_per_class)
        torch.save(keyword, keyword_path)

    masked_dataset = MaskedDataset(dataset, keyword)

    return masked_dataset


class Keyword(object):
    def __init__(self, keyword_type, keyword):
        self.keyword_type = keyword_type
        self.keyword = keyword

    def __len__(self):
        return len(self.keyword)


def get_keyword(args, dataset, tokenizer, keyword_type, keyword_per_class):
    if keyword_type == 'tfidf':
        keyword = get_tfidf_keyword(dataset, keyword_per_class)
        keyword = Keyword('tfidf', keyword)

    elif keyword_type == 'attention':
        if args.attn_backbone is None:
            args.attn_backbone = args.backbone

        attn_model, _ = load_backbone(args.attn_backbone, output_attentions=True)
        attn_model.to(device)  # only backbone

        assert args.attn_model_path is not None
        state_dict = torch.load(os.path.join(CKPT_PATH, dataset.data_name, args.attn_model_path))

        new_state_dict = dict()
        for key, value in state_dict.items():  # only keep backbone parameters
            if key.split('.')[0] == 'backbone':
                key = '.'.join(key.split('.')[1:])  # remove 'backbone'
                new_state_dict[key] = value

        attn_model.load_state_dict(new_state_dict)  # backbone state dict

        if torch.cuda.device_count() > 1:
            attn_model = nn.DataParallel(attn_model)

        keyword = get_attention_keyword(dataset, attn_model, keyword_per_class)
        keyword = Keyword('attention', keyword)

    else:  # random
        keyword = list(tokenizer.vocab.values())  # all words
        keyword = Keyword('random', keyword)

    return keyword


def get_tfidf_keyword(dataset, keyword_per_class=10):
    from sklearn.feature_extraction.text import TfidfVectorizer

    SPECIAL_TOKENS = dataset.tokenizer.all_special_ids

    class_docs = [''] * dataset.n_classes  # concat all texts for each class

    raw_texts = dataset._load_dataset('train', raw_text=True)
    for (text, label) in raw_texts:
        class_docs[label] += text

    tfidf = TfidfVectorizer(ngram_range=(1, 1))
    feat = tfidf.fit_transform(class_docs).todense()  # (n_classes, vocabs)
    feat = np.squeeze(np.asarray(feat))  # matrix -> array

    keyword = []
    for cls in range(dataset.n_classes):
        sorted_idx = feat[cls].argsort()[::-1]

        count = 0
        for idx in sorted_idx:
            if count == keyword_per_class:
                break

            word = tfidf.get_feature_names()[idx]
            token = dataset.tokenizer.encode(word)[1:-1]  # ignore CLS and SEP

            if token in SPECIAL_TOKENS:  # special token
                continue
            elif len(token) > 1:  # multiple words
                continue

            if token not in keyword:
                 keyword.append(token)
                 count += 1

    assert len(keyword) == keyword_per_class * dataset.n_classes

    return keyword


def get_attention_keyword(dataset, attn_model, keyword_per_class=10):
    loader = DataLoader(dataset.train_dataset, shuffle=False,
                        batch_size=16, num_workers=4)

    SPECIAL_TOKENS = dataset.tokenizer.all_special_ids
    PAD_TOKEN = dataset.tokenizer.convert_tokens_to_ids(dataset.tokenizer.pad_token)

    vocab_size = len(dataset.tokenizer)

    attn_score = torch.zeros(vocab_size)
    attn_freq = torch.zeros(vocab_size)

    for _, (tokens, _) in enumerate(loader):
        tokens = tokens.to(device)

        with torch.no_grad():
            out_h, out_p, attention_layers = attn_model(tokens)

        attention = attention_layers[-1]  # attention of final layer (batch_size, num_heads, max_len, max_len)
        attention = attention.sum(dim=1)  # sum over attention heads (batch_size, max_len, max_len)

        for i in range(attention.size(0)):  # batch_size
            for j in range(attention.size(-1)):  # max_len
                token = tokens[i][j].item()

                if token == PAD_TOKEN: # token == pad_token
                    break
                
                if token in SPECIAL_TOKENS:  # skip special token
                    continue

                score = attention[i][0][j]  # 1st token = CLS token

                attn_score[token] += score.item()
                attn_freq[token] += 1

    for tok in range(vocab_size):
        if attn_freq[tok] == 0:
            attn_score[tok] = 0
        else:
            attn_score[tok] /= attn_freq[tok]  # normalize by frequency

    num = keyword_per_class * dataset.n_classes  # number of total keywords
    keyword = attn_score.argsort(descending=True)[:num].tolist()

    return keyword

