import torch
import itertools
from decompositions import index_sel
from tqdm import tqdm
from grad_dist import calnorm
import numpy as np
import math
import gc


def process_indices(indices):
    '''
    Processes indices to generate a list of cumulative indices
    '''

    l2 = indices[0]
    for i in range(len(indices) - 1):
        l2 = l2 + list(np.array(l2[-1]) + np.array(indices[i + 1]))

    return l2


def sample_selection(trainloader, data3, net, clone_dict, batch_size, fraction, sel_iter, numEpochs, device, dataset_name):
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    if dataset_name.lower() == 'boston':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    else:
        loss_fn = torch.nn.functional.cross_entropy
    assert numEpochs > sel_iter, "Number of Epochs must be greater than sel_iter"
    indices = []
    l2 = []    
    len_ranks = batch_size * fraction
    min_range = int(len_ranks - (len_ranks * fraction))
    max_range = int(len_ranks + (len_ranks * fraction))
    
    if max_range - min_range < 1:
        ranks = np.arange((1, max_range),1, dtype=int)
        num_selections = int(numEpochs / sel_iter)
        candidates = ranks
    else:    
        ranks = np.arange(min_range, max_range, 1, dtype=int)
        num_selections = int(numEpochs / sel_iter)
        candidates = math.ceil(len(ranks) / num_selections)
    
    candidate_ranks = list(np.random.choice(list(ranks), size=candidates, replace=False))
    if len(candidate_ranks) > 3:
        candidate_ranks = list(np.random.choice(list(candidate_ranks), size=3, replace=False))
    print("current selected rank candidates:", candidate_ranks)

    
    # Add success status tracking
    total_samples = len(trainloader.dataset)
    selected_count = 0
    success_rate = 0.0
    
    for _, ((trainsamples, labels), V) in enumerate(tqdm(zip(trainloader, data3), desc="Sample Selection")):
        
        net.load_state_dict(clone_dict)
        trainsamples = trainsamples.to(device)
        labels = labels.to(device)
        
        
        A = np.reshape(trainsamples.detach().cpu().numpy(),(-1,trainsamples.shape[0]))
        out, _ = net(trainsamples, last=True, freeze=True)
        
        
        loss = loss_fn(out, labels).sum()
        l0_grad = torch.autograd.grad(loss, out)[0]
        distance_dict = {}
        for ranks in candidate_ranks:
            net.load_state_dict(clone_dict)
            idx2 = index_sel(V,  min(ranks, A.shape[1]))
            idx2 = list(set((itertools.chain(*idx2))))
            if dataset_name == "boston":
                out_idx, _ = net(trainsamples[idx2,:], last=True, freeze=True)
            else:
                out_idx, _ = net(trainsamples[idx2,:,:,:], last=True, freeze=True)
            loss_idx = loss_fn(out_idx, labels[idx2]).sum()
            l0_idx_grad = torch.autograd.grad(loss_idx, out_idx)[0]
            distance = calnorm(l0_idx_grad, l0_grad)
            distance_dict[tuple(idx2)] = distance 
    
        indices.append(list(min(distance_dict, key=distance_dict.get)))
        selected_count += len(idx2)
        success_rate = (selected_count / total_samples) * 100
        
    print(f"Sample Selection Complete - Selected {selected_count}/{total_samples} samples ({success_rate:.2f}%)")
    
    del clone_dict
    del net
    torch.cuda.empty_cache()    
    gc.collect()

    # Process collected indices
    batch_indices = []
    total_indices = []
    
    for batch_idx in indices:
        if isinstance(batch_idx, list):
            batch_indices.extend(batch_idx)
        else:
            batch_indices.append(batch_idx)
    
    # Convert to occurrence count/scores
    unique_indices = np.unique(batch_indices)
    scores = np.zeros(len(trainloader.dataset))
    for idx in batch_indices:
        scores[idx] += 1
    
    # Select top fraction based on scores
    num_to_select = int(len(trainloader.dataset) * fraction)
    selected_indices = np.argsort(scores)[::-1][:num_to_select]
    
    # Ensure we have exactly the right number of unique indices
    selected_indices = np.unique(selected_indices)[:num_to_select]
    
    final_success_rate = (len(selected_indices) / len(trainloader.dataset)) * 100
    print(f"Final Selection - Kept {len(selected_indices)}/{len(trainloader.dataset)} samples ({final_success_rate:.2f}%)")
    
    return selected_indices

