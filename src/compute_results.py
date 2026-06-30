import json
import numpy as np
import torch
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import datetime

from rerank_utils import (
    build_frames_provider_webvid,
    qwen2vl_text_rerank
)


def aggregation_and_retrieval(
    ref_img_features: torch.Tensor,  
    query_feats_t_enrich: torch.Tensor, 
    index_features: torch.Tensor,  
    tau: float = 0.04,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   
    S_r = torch.matmul(ref_img_features, index_features.T)  # [n, tar_num]
    S_E = torch.matmul(query_feats_t_enrich, index_features.T) 
    M = index_features.shape[0]
    score_i = (torch.relu(S_r.unsqueeze(1) - S_E) / M).sum(dim=-1)

    weights = F.softmax(score_i / tau, dim=-1)  # [n, NUM_CAP]
    ensemble_feats = (weights.unsqueeze(-1) * query_feats_t_enrich).sum(dim=1)  # [n, clip_dim]

    return ensemble_feats


def get_time()->str:
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d-%H_%M_%S")    
    
@torch.no_grad()
def webvidcovr(
    device: torch.device,
    ref_img_features: torch.Tensor,
    query_feats_v: torch.Tensor,
    query_feats_t: torch.Tensor,
    query_feats_t_enrich: torch.Tensor,
    tar_cap_feats: torch.Tensor,
    tar_captions: list,
    target_names: list,
    reference_names: list,
    index_features: torch.Tensor,
    index_names: list,
    query_ids: list,
    dataset_name: str,
    dataset_path: str,
    task: str,
    split: str = 'test',
    lam: float = 0.2,
    frames_provider=None,
    rerank_method: str = 'None',
    **kwargs
) -> tuple:
    """
    Compute retrieval metrics for WebVidCoVR dataset
    """

    ref_img_features = torch.nn.functional.normalize(ref_img_features, dim=-1).to(device)
    query_feats_v = torch.nn.functional.normalize(query_feats_v, dim=-1).to(device)
    query_feats_t = torch.nn.functional.normalize(query_feats_t, dim=-1).to(device)
    index_features = torch.nn.functional.normalize(index_features, dim=-1).to(device)
    query_feats_t_enrich = torch.nn.functional.normalize(query_feats_t_enrich, dim=-1).to(device)

    if query_feats_t_enrich.dim() > 2:
        ensemble_feats = aggregation_and_retrieval(
            ref_img_features=ref_img_features,
            query_feats_t_enrich=query_feats_t_enrich,
            index_features=index_features,
            device=device
        )
        ensemble_feats = torch.nn.functional.normalize(ensemble_feats, dim=-1).to(device)
    else:
        ensemble_feats = query_feats_t_enrich

    # Compute similarities
    sim_v_t = query_feats_v @ tar_cap_feats.T
    sim_t_v = ((ensemble_feats + query_feats_t) / 2) @ index_features.T

    sim_b = sim_v_t * sim_t_v
    similarities = lam * sim_t_v + (1 - lam) * sim_b
    similarities = similarities.cpu().numpy()

    rerank_similarities = similarities.copy()

    for i in range(len(reference_names)):
        for j in range(len(target_names)):
            if str(reference_names[i]) == str(target_names[j]):
                similarities[i][j] = -10
    
    recalls = eval_recall(similarities)
    print(recalls)

    if rerank_method != 'None':
        u_stats = compute_uncertainty(rerank_similarities, device=device)
        uncertainty = u_stats["uncertainty"]
    
        frames_provider = build_frames_provider_webvid(
            video_base_dir="Your vid_dirs path",
            video_extension="mp4",
            max_frames=8,
            temp_dir=f'{dataset_path}/task/{task}/rerank_ref_frames/',
        )

        instructions = kwargs.get('instructions', [])

        clip_name = kwargs.get('clip', 'no_clip')
        cache_path = f'{dataset_path}/task/{task}/rerank_caches/{clip_name}_rerank_cache.pkl'
        similarities_reranked = qwen2vl_text_rerank(
            cache_path=cache_path,
            uncertainty=uncertainty,
            similarities=rerank_similarities,
            device=device,
            tar_captions=tar_captions,
            reference_names=reference_names,
            frames_provider=frames_provider,
            instructions=instructions,
            strategy=kwargs.get('qwen_rerank_strategy', 'boost_max'),
            topk=20,   
            u_topk=1000,              
            lam=0.2       
        )


        for i in range(len(reference_names)):
            for j in range(len(target_names)):
                if str(reference_names[i]) == str(target_names[j]):
                    similarities_reranked[i][j] = -10
        print(f"\nRe-rank: ")
        recalls = eval_recall(similarities_reranked)
        print(recalls)
    
    return recalls

@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks) 
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    mnr = np.mean(ranks)
    mdr = np.median(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "MnR": round(mnr, 2),  
        "MdR": round(mdr, 2), 
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result


def compute_uncertainty(similarities, device, tau: float = 0.02, alpha: float = 0.5):
    sim_tensor = torch.tensor(similarities, device=device)  
    # ---- Step 1: Softmax  ----
    p = F.softmax(sim_tensor / tau, dim=-1)  # [N, M]
    
    # ---- Step 2: Entropy calculation ----
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)  # [N]

    # ---- Step 3: Top-2 Margin ----
    top2 = torch.topk(sim_tensor, k=2, dim=-1).values    # [N, 2]
    margin = top2[:, 1] / (top2[:, 0] + 1e-8)        
    margin = torch.clamp(margin, 0, 1)

    # ---- Step 4: Standardized entropy and margin ----
    entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
    margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)

    # ---- Step 5: comprehensive indicator ----
    uncertainty = alpha * entropy_norm + (1 - alpha) * margin_norm
    confidence = 1 - uncertainty

    return {
        "entropy": entropy,
        "margin": margin,
        "uncertainty": uncertainty,
        "confidence": confidence
    }
