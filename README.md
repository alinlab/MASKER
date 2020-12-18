# MASKER: Masked Keyword Regularization for Reliable Text Classification

Official PyTorch implementation of ["**MASKER: Masked Keyword Regularization for Reliable Text Classification**"](https://arxiv.org/abs/2012.09392) (AAAI 2021)
by [Seung Jun Moon*](https://github.com/SeungJunnn), [Sangwoo Mo*](https://sites.google.com/view/sangwoomo), [Kimin Lee](https://sites.google.com/view/kiminlee),
[Jaeho Lee](https://jaeho-lee.github.io/), and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html).

## Setup

### Download datasets

Download datasets from [Google Drive](https://drive.google.com/file/d/19Y3qgBosuysAaQtV5SFIxDfwxA7DztSr/view?usp=sharing) and locate files in `./dataset`.

Set `DATA_PATH` (default: `./dataset`) and `CKPT_PATH` (default: `./checkpoint`) from `common.py`.
Datafiles should be located in the corresponding directory `DATA_PATH/{data_name}`.
For example, IMDB datafiles should be located in `DATA_PATH/imdb/imdb.txt`.

The dataset will be pre-processed into a TensorDataset and be saved in
```
DATA_PATH/{data_name}/{base_path}.pth
```
where `base_path = "{data_name}_{model_name}_{suffix}"` 
and suffix indicates split ratio, random seed, train/test, etc.

### Generate keywords

One needs pre-computed keywords to train [residual ensemble](#train-residual-ensemble) or [MASKER](#train-masker).

When running such models, the keywords will be automatically saved in
```
DATA_PATH/{data_name}/{base_path}_keyword_{keyword_type}_{keyword_per_class}.pth
```
and the biased/masked dataset will be saved in
```
DATA_PATH/{data_name}/{base_path}_{biased/masked}_{keyword_type}_{keyword_per_class}.pth
```


## Train models

### Train vanilla BERT

Train a vanilla BERT model. The model will be saved in `review_bert-base-uncased_sub_0.25_seed_0.model`.\
One need to train vanilla BERT first to get attention keywords for residual ensemble and MASKER models.
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --train_type base \
    --backbone bert --classifier_type softmax --optimizer adam_ood \
```

### Train residual ensemble

Train a keyword biased model. Need to specify the `attn_model_path` for attention keywords.
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --train_type base --use_biased_dataset \
    --backbone bert --classifier_type softmax --optimizer adam_ood \
    --attn_model_path review_bert-base-uncased_sub_0.25_seed_0.model
```

Train a residual ensemble [1,2] model. Need to specify the `biased_model_path`.
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --train_type residual \
    --backbone bert --classifier_type softmax --optimizer adam_ood \
    --biased_model_path review_bert-base-uncased_sub_0.25_seed_0_biased.model
```

[1] Clark et al. Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases. EMNLP 2019. \
[2] He et al. Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual. EMNLP Workshop 2019.

### Train MASKER

Train a MASKER model. Need to specify the `attn_model_path` for attention keywords.
```
python train.py --dataset review --split_ratio 0.25 --seed 0 \
    --train_type masker \
    --backbone bert --classifier_type sigmoid --optimizer adam_ood \
    --keyword_type attention --lambda_ssl 0.001 --lambda_ent 0.0001 \
    --attn_model_path review_bert-base-uncased_sub_0.25_seed_0.model
```


## Evalaute models

### Evaluate classification

Specify `test_dataset` for domain generalization results (in-distribution if not specified).
```
python eval.py --dataset review --split_ratio 0.25 --seed 0 \
    --eval_type acc --test_dataset review \
    --backbone bert --classifier_type softmax \
    --model_path review_bert-base-uncased_sub_0.25_seed_0.model
```

### Evaluate OOD detection

Specify `ood_datasets` for OOD detection results.
```
python eval.py --dataset review --split_ratio 0.25 --seed 0 \
    --eval_type ood --ood_datasets remain \
    --backbone bert --classifier_type softmax \
    --model_path review_bert-base-uncased_sub_0.25_seed_0.model
```

