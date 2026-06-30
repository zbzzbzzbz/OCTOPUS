import json
import os
from typing import Optional, Tuple, List, Dict, Union
import datetime
import pandas as pd
import clip
import numpy as np
import pickle
import torch
import tqdm
import data_utils
from classes import Arch
import csv
if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


def get_time()->str:
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d-%H_%M_%S")
        
def compute_query_feats_v(f_p, f_q, eps=1e-8):
    device = f_p.device
    f_q = f_q.to(device)
    eps_tensor = torch.tensor(eps, device=device) 
    
    max_f_p = f_p.max(dim=1, keepdim=True)[0]  
    max_f_q = f_q.max(dim=1, keepdim=True)[0]  
    
    weight_q = max_f_p / (max_f_q + eps_tensor) 
    f_rp = f_p + weight_q * f_q

    return f_rp


@torch.no_grad()
def extract_gen_image_features(device: torch.device, dataset_name: str, dataset: torch.utils.data.Dataset, clip_model, batch_size: Optional[int] = 4,
                           num_workers: Optional[int] = 0, preload: str=None,arch= None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    print(f"preload : {preload}")
    
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        gen_tar_features, gen_reference_names = extracted_data['gen_tar_features'], extracted_data['gen_reference_names']
        print(f"len(gen_tar_features): {len(gen_tar_features)}")
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=data_utils.collate_fn)

        gen_tar_features, gen_reference_names, = [], []

        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features    
        for batch in tqdm.tqdm(loader):
            gen_images = batch['gen_tar_img'].squeeze(1)
            gen_batch_names = batch['reference_name']
        
            gen_images = gen_images.to(device)

            with torch.no_grad(),torch.amp.autocast(device_type="cuda"):
                gen_batch_features = clip_model.encode_image(gen_images)
                    
                gen_tar_features.append(gen_batch_features.cpu())
                gen_reference_names.extend(gen_batch_names)

        gen_tar_features = torch.vstack(gen_tar_features)
        print(f"len(gen_tar_features): {len(gen_tar_features)}")

        if preload is not None:
            os.makedirs(os.path.dirname(preload), exist_ok=True)
            pickle.dump({'gen_tar_features': gen_tar_features, 'gen_reference_names': gen_reference_names}, open(preload, 'wb'))
            print(f"Save gen_tar image feathers in {preload}")
    return gen_tar_features, gen_reference_names

@torch.no_grad()
def extract_ref_image_features(device: torch.device, dataset_name: str, dataset: torch.utils.data.Dataset, clip_model, batch_size: Optional[int] = 4,
                           num_workers: Optional[int] = 0, preload: str=None,arch= None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    print(f"preload : {preload}")
    
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        ref_features, reference_names = extracted_data['ref_features'], extracted_data['reference_names']
        print(f"len(ref_features): {len(ref_features)}")
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=data_utils.collate_fn)

        ref_features, ref_names, = [], []

        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features    
        for batch in tqdm.tqdm(loader):
            ref_images = batch['reference_image'].squeeze(1)
            reference_names = batch['reference_name']
        
            ref_images = ref_images.to(device)

            if ref_images.dim() == 5:
                # print(f"ref_images.dim(): {ref_images.dim()}")
                b, f, c, h, w = ref_images.shape
                ref_images = ref_images.view(b * f, c, h, w)

                with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                    ref_batch_features = clip_model.encode_image(ref_images)
                    ref_batch_features = ref_batch_features.view(b, f, -1).mean(dim=1)
            else:
                with torch.no_grad(),torch.amp.autocast(device_type="cuda"):
                    ref_batch_features = clip_model.encode_image(ref_images)
                    
            ref_features.append(ref_batch_features.cpu())
            ref_names.extend(reference_names)

        ref_features = torch.vstack(ref_features)
        print(f"ref_features.shape: {ref_features.shape}")

        if preload is not None:
            os.makedirs(os.path.dirname(preload), exist_ok=True)
            pickle.dump({'ref_features': ref_features, 'reference_names': reference_names}, open(preload, 'wb'))
            print(f"Save ref and gen_tar image feathers in {preload}")
    return ref_features, reference_names

@torch.no_grad()
def extract_tar_image_features(device: torch.device, dataset_name: str, dataset: torch.utils.data.Dataset, clip_model, batch_size: Optional[int] = 4,
                           num_workers: Optional[int] = 0, preload: str=None,arch= None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    print(f"preload : {preload}")
    
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        print(f"len(index_features): {len(index_features)}")
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=data_utils.collate_fn)

        index_features, index_names = [], []

        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features    
        for batch in tqdm.tqdm(loader):
            images = batch['image'].squeeze(1)
            names = batch['image_name']
            images = images.to(device)

            if images.dim() == 5:
                # print(f"images.dim(): {images.dim()}")
                b, f, c, h, w = images.shape
                images = images.view(b * f, c, h, w)

                with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                    batch_features = clip_model.encode_image(images)  # [b*f, d]
                    batch_features = batch_features.view(b, f, -1).mean(dim=1)  # [b, d] 
            else:
                with torch.no_grad(),torch.amp.autocast(device_type="cuda"):
                    batch_features = clip_model.encode_image(images)
                        
            index_features.append(batch_features.cpu())
            index_names.extend(names)

        index_features = torch.vstack(index_features)
        print(f"len(index_features): {len(index_features)}")

        if preload is not None:
            os.makedirs(os.path.dirname(preload), exist_ok=True)
            pickle.dump({'index_features': index_features, 'index_names': index_names}, open(preload, 'wb'))
            print(f"Save image feathers in {preload}")
    return index_features, index_names


@torch.no_grad()
def generate_predictions(
    device: torch.device, dataset_name:str,clip_model: clip.model.CLIP,query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]],arch:Arch,
    dataset_path,compute_results_function,index_features,index_names,ref_features,gen_tar_features,reference_names,task,split,**kwargs
) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions
    """    
    ### Generate BLIP Image Captions.
    torch.cuda.empty_cache()    
    batch_size = 4
    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        ref_captions, relative_captions, tar_captions = [], [], []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []
        
        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=4, 
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Generating image captions...')
        
        for batch in query_iterator:
            if 'ref_caption' in batch:
                ref_captions.extend(batch['ref_caption'])

            reference_names.extend(batch['reference_name'])
            if 'fashioniq' not in dataset_name:
                relative_captions.extend(batch['relative_caption'])
            else:
                rel_caps = batch['relative_caption']
                rel_caps = np.array(rel_caps).T.flatten().tolist()
                relative_captions.extend([
                    f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)
                    ])

            if 'target_caption' in batch:
                tar_captions.extend(batch['target_caption'])
            if 'target_name' in batch:
                target_names.extend(batch['target_name'])

            gt_key = 'gt_img_ids'
            if 'group_members' in batch:
                gt_key = 'group_members'
            if gt_key in batch:
                gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

            query_key = 'query_id'
            if 'pair_id' in batch:
                query_key = 'pair_id'
            if query_key in batch:
                query_ids.extend(batch[query_key])
     
                
        if preload_dict['captions'] is not None:
            os.makedirs(os.path.dirname(preload_dict['captions']), exist_ok=True)
            res_dict = {
                'ref_captions': ref_captions, 
                'gt_img_ids': gt_img_ids, 
                'relative_captions': relative_captions,
                'target_names': target_names,
                'reference_names': reference_names,
                'query_ids': query_ids,
                'tar_captions': tar_captions
            }
            pickle.dump(res_dict, open(preload_dict['captions'], 'wb'))
    else:
        print(f'Loading precomputed image captions from {preload_dict["captions"]}!')
        
        # load captions from pickle file
        try:
            with open(preload_dict['captions'], 'rb') as f:
                caption_data = pickle.load(f)
                ref_captions = caption_data['ref_captions'] if 'ref_captions' in caption_data else []
                gt_img_ids = caption_data['gt_img_ids'] if 'gt_img_ids' in caption_data else []
                relative_captions = caption_data['relative_captions']
                tar_captions = caption_data['tar_captions'] if 'tar_captions' in caption_data else []
                target_names = caption_data['target_names']
                reference_names = caption_data['reference_names']
                query_ids = caption_data['query_ids']
            print(f"Successfully loaded precomputed captions data")
        except (FileNotFoundError, pickle.PickleError, KeyError) as e:
            print(f"Loading pickle file failed: {e}")
            raise ValueError("Unable to read caption file, please delete file and regenerate")         

    
    print("Loading image caption done")  

    if dataset_name.lower() in {'webvidcovr'}:
        df = pd.read_csv(f"{dataset_path}/preload/gen_tar_desc/enriched_target_desc.csv")
    
        df['target'] = df['target'].astype(str)
        filtered_df  = df[df['target'].isin(target_names)]
        filtered_df = filtered_df.set_index('target').loc[target_names].reset_index()
        modified_captions_enrich = [json.loads(s) for s in filtered_df['enriched_desc'].tolist()]
        modified_captions = filtered_df['tar_desc'].tolist()
                
    print("Loading precomputed caption modifiers from... done")  
    
    LLM_tar_cap_feats = text_encoding(device, clip_model, modified_captions, batch_size=batch_size,arch=arch)

    NUM_CAP = 10

    if isinstance(modified_captions_enrich[0], list):
        fixed_captions = []
        for sublist in modified_captions_enrich:
            current_len = len(sublist)
            if current_len < NUM_CAP:
                print(f"len(sublist) : {current_len}")
                # Take the last element as the fill value
                fill_value = sublist[-1] if current_len > 0 else "" 
                # Calculate the quantity that needs to be supplemented and fill in
                sublist += [fill_value] * (NUM_CAP - current_len)
            fixed_captions.append(sublist[:NUM_CAP])

        flat_captions = [cap for sublist in fixed_captions for cap in sublist]
        print(f"len(flat_captions): {len(flat_captions)}")

        # Batch encode all descriptions
        flat_features = text_encoding(
            device, clip_model, 
            flat_captions, 
            batch_size=batch_size, 
            arch=arch
        )

        # Reshaping into [N, NUM_CP, dim] tensors
        N = len(modified_captions_enrich)
        dim = flat_features.shape[1]
        LLM_tar_cap_enrich_feats = flat_features.view(N, NUM_CAP, dim)
        print(f"LLM_tar_cap_enrich_feats.shape: {LLM_tar_cap_enrich_feats.shape}")
        assert LLM_tar_cap_enrich_feats.shape == (N, NUM_CAP, dim), f"Shape error: {LLM_tar_cap_enrich_feats.shape}"
    else:
        LLM_tar_cap_enrich_feats = text_encoding(device, clip_model, modified_captions_enrich, batch_size=batch_size, arch=arch)


    if len(tar_captions) > 0:
        tar_cap_feats = text_encoding(device, clip_model, tar_captions, batch_size=batch_size, arch=arch)
    else:
        tar_cap_feats = None

    # query_feats_v
    query_feats_v = compute_query_feats_v(f_p=gen_tar_features, f_q=ref_features)
    
    return {
        'gen_tar_features': gen_tar_features,
        'ref_img_features': ref_features,
        'query_feats_v': query_feats_v, 
        'query_feats_t': LLM_tar_cap_feats,
        'query_feats_t_enrich': LLM_tar_cap_enrich_feats,
        'tar_cap_feats': tar_cap_feats,
        'index_features': index_features,
        'target_names': target_names, 
        'reference_names': reference_names,
        'query_ids': query_ids,
        'tar_captions': tar_captions,
        'instructions': relative_captions,
        'gt_img_ids': gt_img_ids,
    }



def text_encoding(device, clip_model,input_captions, arch:Arch,batch_size=32):
    n_iter = int(np.ceil(len(input_captions)/batch_size))
    predicted_features = []
    
    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
        if hasattr(clip_model, 'tokenizer'):
            tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
        else:
            tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)
        clip_text_features = clip_model.encode_text(tokenized_input_captions).float()
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)        
        
    return torch.nn.functional.normalize(predicted_features, dim=-1)
    