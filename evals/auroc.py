import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_aurocs(model, id_loader, ood_loaders, classifier_type='softmax'):
    print('Compute AUROCs...')
    model.eval()

    aurocs = {ood_name: 0 for ood_name in ood_loaders.keys()}

    scores_id = get_scores(model, id_loader, classifier_type)
    for ood_name, ood_loader in ood_loaders.items():
        scores_ood = get_scores(model, ood_loader, classifier_type)
        aurocs[ood_name] = get_auroc(scores_id, scores_ood)

    return aurocs


def get_scores(model, loader, classifier_type='softmax'):
    assert classifier_type in ['softmax', 'sigmoid']

    scores_all = []
    for i, (tokens, _) in enumerate(loader):
        tokens = tokens.to(device)

        with torch.no_grad():
            outputs = model(tokens)  # (B, C)

        if classifier_type == 'softmax':
            outputs = F.softmax(outputs)
        else:
            outputs = torch.sigmoid(outputs)

        scores = outputs.max(dim=1)[0]
        scores_all.append(scores.cpu().numpy())

    return np.concatenate(scores_all)


def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores) * 100

