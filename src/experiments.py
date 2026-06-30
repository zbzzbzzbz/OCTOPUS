from argparse import Namespace
import torch
import os
import utils
from datasets import WebVidCoVRDataset, WebVidCoVRDatasetDome
import compute_results
import clip
from transformers import CLIPProcessor, CLIPModel
import termcolor
from classes import Captioner,Arch

class Experiment:
    def __init__(self,args:Namespace) -> None:
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)
        self.device = torch.device(f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu')
        
    def run(self):
        clip_model,clip_processor = self.load_Clip_model()
        target_datasets,query_datasets,compute_results_function,pairings = self.load_dataset(clip_processor)
        self.evaluate(query_datasets, target_datasets, pairings,compute_results_function,clip_model,clip_processor)
        print("Finish.")
        return
    
    def data_preprocessing(self):
        return
    def load_Clip_model(self):
        clip_device = self.device
        self.arch = Arch.from_string(self.clip)
        clip_model,clip_processor = self.arch.load_model_and_preprocess(device=clip_device)
        print('Done.')
        return clip_model,clip_processor

    def load_dataset(self,clip_processor:CLIPProcessor):
        ### Load Evaluation Datasets.
        target_datasets, query_datasets, pairings = [], [], []
        
        if self.dataset.lower() == 'webvidcovr':
            target_datasets.append(WebVidCoVRDatasetDome(
            # target_datasets.append(WebVidCoVRDataset(
                transform=clip_processor,
                annotation=self.annotation,
                vid_dir=self.vid_dirs,
                gen_tar_dir=self.gen_tar_dirs,
                split=self.split,
                vid_query_method=self.vid_query_method if hasattr(self, 'vid_query_method') else 'middle',
                vid_frames=self.vid_frames if hasattr(self, 'vid_frames') else 1,
                mode='classic'
            ))
            query_datasets.append(WebVidCoVRDatasetDome(
            # query_datasets.append(WebVidCoVRDataset(
                transform=clip_processor,
                annotation=self.annotation,
                vid_dir=self.vid_dirs,
                gen_tar_dir=self.gen_tar_dirs,
                split=self.split,
                vid_query_method=self.vid_query_method if hasattr(self, 'vid_query_method') else 'middle',
                vid_frames=self.vid_frames if hasattr(self, 'vid_frames') else 1,
                mode='relative'
            ))
            compute_results_function = compute_results.webvidcovr
            pairings.append('default')
            assert len(target_datasets) > 0, f"target_datasets is none"
            print("load_dataset webvidcovr done") 
        else:
            raise ValueError(f"dataset_name error !!")
            
        return target_datasets,query_datasets,compute_results_function,pairings

    def evaluate(self,query_datasets, target_datasets, pairings,compute_results_function,clip_model,clip_processor):
        preload_dict = {key: None for key in ['img_features', 'captions', 'ref_img_features', 'gen_img_features']}
        if 'captions' in self.preload:
            preload_dict['captions'] = f'{self.dataset_path}/preload/image_captions/{self.preload_image_captions_file}'
        if 'img_features' in self.preload:
            preload_dict['img_features'] = f'{self.dataset_path}/preload/img_features/{self.clip}_{self.dataset}_{self.split}.pkl'
        if 'ref_img_features' in self.preload:
            preload_dict['ref_img_features'] = f'{self.dataset_path}/preload/ref_img_features/{self.clip}_{self.dataset}_{self.split}.pkl'
        if 'gen_img_features' in self.preload:
            preload_dict['gen_img_features'] = f'{self.dataset_path}/preload/gen_img_features/{self.clip}_{self.dataset}_{self.split}.pkl'


        for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
            termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])
            
            ### General Input Arguments.
            input_kwargs = {
                'dataset_name':self.dataset,
                'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model, 
                'processor': clip_processor, 'device': self.device, 'split': self.split,
                'preload_dict': preload_dict,'arch':self.arch,
                'clip':self.clip,'dataset_path':self.dataset_path,'compute_results_function':compute_results_function,
                "task":self.task
            }    
            
            ### Compute Target Image Features
            print(f'Extracting target image features using CLIP: {self.clip}.')
            index_features, index_names = utils.extract_tar_image_features(
                self.device, self.dataset, target_dataset, clip_model, preload=preload_dict['img_features'],arch = self.arch)
            index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
            input_kwargs.update({'index_features': index_features, 'index_names': index_names})
            print("Extracting target image features done")

            ### Compute Ref Image Features
            print(f'Extracting ref image features using CLIP: {self.clip}.')
            ref_features, reference_names = utils.extract_ref_image_features(
                self.device, self.dataset, query_dataset, clip_model, preload=preload_dict['ref_img_features'],arch = self.arch)
            ref_features = torch.nn.functional.normalize(ref_features.float(), dim=-1)
            input_kwargs.update({'ref_features': ref_features, 'reference_names': reference_names})
            print("Extracting ref image features done")

            ### Compute Gen Image Features
            print(f'Extracting gen_tar image features using CLIP: {self.clip}.')
            gen_tar_features, gen_reference_names = utils.extract_gen_image_features(
                self.device, self.dataset, query_dataset, clip_model, preload=preload_dict['gen_img_features'],arch = self.arch)
            gen_tar_features = torch.nn.functional.normalize(gen_tar_features.float(), dim=-1)
            input_kwargs.update({'gen_tar_features': gen_tar_features})
            print("Extracting gen_tar image features done")

            ### Compute Method-specific Query Features.
            print(f'Generating conditional query predictions (CLIP: {self.clip}).')
            out_dict = utils.generate_predictions(**input_kwargs)
            input_kwargs.update(out_dict)
            print("Generating conditional query predictions done")
                        
            ### Compute Dataset-specific Retrieval Scores.
            print('Computing final retrieval metrics.')
            input_kwargs.update(out_dict)                    
            compute_results_function(**input_kwargs)    
                     
