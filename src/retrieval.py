import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  

def select_subset_indices(dataset, subset_file, subset_size=1000):
    import os
    import random

    if os.path.exists(subset_file):
        with open(subset_file, 'r') as f:
            indices = json.load(f)  # list of ints
        print(f"Loaded {len(indices)} subset indices from {subset_file}")
        return indices
    else:
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        subset = all_indices[:subset_size]
        with open(subset_file, 'w') as f:
            json.dump(subset, f)
        print(f"Created new subset of size {subset_size} and wrote to {subset_file}")
        return subset

def compute_recall_at_k(sim_matrix):
    """
    Given NxN sim_matrix, where sim_matrix[i,j] is the similarity
    of query i to item j, the correct match is j=i.
    R@1, R@5, R@10, R@20.

    Returns: {'r1':..., 'r5':..., 'r10':..., 'r20':...}
    """
    N = sim_matrix.shape[0]
    ranks = []
    for i in range(N):
        row = sim_matrix[i]
        sorted_indices = np.argsort(-row)
        rank_of_correct = np.where(sorted_indices == i)[0][0]  # 0-based
        ranks.append(rank_of_correct)
    ranks = np.array(ranks)
    r1  = np.mean(ranks < 1)
    r5  = np.mean(ranks < 5)
    r10 = np.mean(ranks < 10)
    r20 = np.mean(ranks < 20)
    return {
        'r1':  r1,
        'r5':  r5,
        'r10': r10,
        'r20': r20
    }


# For Text->Visual retrieval:
def aggregator_tv_t2v(t_feats, v_feats, temperature):
    token_sims = torch.matmul(t_feats, v_feats.t()) *temperature   
    max_sims = token_sims.max(dim=1).values
    return max_sims.mean().item()

def aggregator_tv_v2t(t_feats, v_feats, temperature):
    token_sims = torch.matmul(t_feats, v_feats.t()) * temperature
    max_sims = token_sims.max(dim=0).values
    return max_sims.mean().item()

def embed_tv_subset(model, dataset, subset_indices, device='cuda', batch_size=8):
    """
    text and image embeddings for the 1000 chosen items in dataset.
    Returns:
        text_feats_list[i]: (Nt_i, D)
        image_feats_list[i]: (Ni_i, D)
    """
    model.eval()
    text_feats_list = [None]*len(subset_indices)
    image_feats_list = [None]*len(subset_indices)

    def collate_tv_eval(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        return images, list(captions)

    class TVSubset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices):
            self.base = base_ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            return self.base.__getitem__(real_idx)

    subset_ds = TVSubset(dataset, subset_indices)
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_tv_eval)

    print(f"Embedding T/V subset of size {len(subset_indices)} ...")
    idx_offset = 0
    with torch.no_grad():
        for batch_images, batch_captions in tqdm(loader, desc="Embedding TV subset"):
            batch_images = batch_images.to(device)
            #batch_images = batch_images.to(dtype=torch.bfloat16)
            vfeats = model.visual_embedder(batch_images)  # (B, Nv, D)
            vfeats = vfeats.to(dtype=torch.float32)

            tfeats, attn_mask = model.text_embedder(batch_captions)  # (B, Nt, D), (B, Nt)
            tfeats = tfeats.to(dtype=torch.float32)
            attn_mask = attn_mask.to(dtype=torch.float32)
            #vfeats = F.normalize(vfeats, dim=1)
            #tfeats = F.normalize(tfeats, dim=1)
            B = vfeats.shape[0]
            for b in range(B):
                # slice out valid tokens
                n_tokens = int(attn_mask[b].sum().item())
                text_feats_list[idx_offset + b] = tfeats[b, :n_tokens].cpu()
                image_feats_list[idx_offset + b] = vfeats[b].cpu()
            idx_offset += B

    return text_feats_list, image_feats_list

def compute_tv_retrieval_metrics(model, dataset, subset_file, device='cuda'):
    """
    1) select subset
    2) embed
    3) build NxN sim matrices for T->V and V->T
    4) compute recall
    5) return dictionary
    """
    indices = select_subset_indices(dataset, subset_file, subset_size=1000)
    text_feats_list, image_feats_list = embed_tv_subset(model, dataset, indices, device=device, batch_size=32)
    N = len(indices)
    scale_factor = torch.exp(model.logit_scale).item()

    # T->V
    print(f"Computing T->V retrieval on {N} items ...")
    sim_mat_t2v = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Aggregator T->V"):
        tfeats_i = text_feats_list[i].to(device)
        for j in range(N):
            vfeats_j = image_feats_list[j].to(device)
            sim_mat_t2v[i, j] = aggregator_tv_t2v(tfeats_i, vfeats_j, scale_factor)
    tv_metrics = compute_recall_at_k(sim_mat_t2v)

    # V->T
    print(f"Computing V->T retrieval on {N} items ...")
    sim_mat_v2t = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Aggregator V->T"):
        vfeats_i = image_feats_list[i].to(device)
        for j in range(N):
            tfeats_j = text_feats_list[j].to(device)
            sim_mat_v2t[i, j] = aggregator_tv_v2t(tfeats_j, vfeats_i, scale_factor)
    vt_metrics = compute_recall_at_k(sim_mat_v2t)

    results = {
        'T->V_r1':  tv_metrics['r1'],
        'T->V_r5':  tv_metrics['r5'],
        'T->V_r10': tv_metrics['r10'],
        'T->V_r20': tv_metrics['r20'],

        'V->T_r1':  vt_metrics['r1'],
        'V->T_r5':  vt_metrics['r5'],
        'V->T_r10': vt_metrics['r10'],
        'V->T_r20': vt_metrics['r20'],
    }
    del text_feats_list, image_feats_list, sim_mat_t2v, sim_mat_v2t
    return results

