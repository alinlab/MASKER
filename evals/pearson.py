import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_pearson(loader, model):
    print('Compute Pearson Correlation...')

    preds = []
    out_label_ids = []
    for i, (tokens, labels) in enumerate(loader):
        tokens = tokens.to(device)

        with torch.no_grad():
            pred = model(tokens)  # (B, C)

        preds.append(pred.cpu().numpy())
        out_label_ids.append(labels.cpu().numpy())

    preds=np.concatenate(preds)
    out_label_ids=np.concatenate(out_label_ids)
    preds=preds.squeeze()
    out_label_ids=out_label_ids.squeeze()

    pearson_corr = pearsonr(preds, out_label_ids)[0]
    spearman_corr = spearmanr(preds, out_label_ids)[0]

    return (pearson_corr + spearman_corr) / 2

