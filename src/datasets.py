import os

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from classes import Arch,Captioner

import pandas as pd
import torch
import cv2
import numpy as np
import random
import csv


def pre_caption(caption, max_words=30):
    return ' '.join(caption.split()[:max_words])

def id2int(x, sub='0'):
    return int(x.replace('/', sub))

def print_dist(msg):
    print(msg)

def write_txt(lines, file):
    with open(file, 'w') as f:
        f.writelines([l + '\n' for l in lines])

class WebVidCoVRDatasetDome(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        gen_tar_dir: str,
        split: str = "test",
        max_words: int = 100,
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        mode: str = "relative",
    ) -> None:
        super().__init__()
        

class WebVidCoVRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        gen_tar_dir: str,
        split: str = "test",
        max_words: int = 100,
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        mode: str = "relative",
    ) -> None:
        super().__init__()

        self.transform = transform
        self.mode = mode
        
        self.annotation_pth = annotation
        assert Path(annotation).exists(), f"Annotation file {annotation} does not exist"
        self.df = pd.read_csv(annotation)
        # self.df: txt1,txt2,sim_txt,pth1,pth2,edit,scores,caption

        self.vid_dir = Path(vid_dir)
        assert self.vid_dir.exists(), f"Video directory {self.vid_dir} does not exist"
        self.gen_tar_dir = Path(gen_tar_dir)
        assert self.gen_tar_dir.exists(), f"Gen Tar directory {self.gen_tar_dir} does not exist"

        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        self.split = split


        vid_pths = self.vid_dir.glob("*/*.mp4")
        gen_tar_pths = self.gen_tar_dir.glob("*/*.jpg")

        id2vidpth = {
            vid_pth.parent.stem + "/" + vid_pth.stem: vid_pth for vid_pth in vid_pths
        }

        id2gentarpth = {
            gen_tar_pth.parent.stem + "/" + gen_tar_pth.stem: gen_tar_pth for gen_tar_pth in gen_tar_pths
        }

        print(f"len(id2vidpth): {len(id2vidpth)}")
        print(f"len(id2gentarpth): {len(id2gentarpth)}")

        assert len(id2vidpth) > 0, f"No videos found in {vid_dir}"
        assert len(id2gentarpth) > 0, f"No videos found in {gen_tar_dir}"

        self.df["path1"] = self.df["pth1"].apply(lambda x: id2vidpth.get(x, None))
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2vidpth.get(x, None))
        self.df["path_gen_tar"] = self.df["pth2"].apply(lambda x: id2gentarpth.get(x, None))

        # Count unique missing paths
        missing_pth1 = self.df[self.df["path1"].isna()]["pth1"].unique().tolist()
        missing_pth1.sort()
        total_pth1 = self.df["pth1"].nunique()

        missing_pth2 = self.df[self.df["path2"].isna()]["pth2"].unique().tolist()
        missing_pth2.sort()
        total_pth2 = self.df["pth2"].nunique()

        missing_pth3 = self.df[self.df["path_gen_tar"].isna()]["pth2"].unique().tolist()
        missing_pth3.sort()
        total_pth3 = self.df["pth2"].nunique()

        if len(missing_pth1) > 0:
            print_dist(
                f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%), saving them to missing_pth1-{split}.txt"
            )
            # write_txt(missing_pth1, f"missing_pth1-{split}.txt")
        if len(missing_pth2) > 0:
            print_dist(
                f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%), saving them to missing_pth2-{split}.txt"
            )
            # write_txt(missing_pth2, f"missing_pth2-{split}.txt")
        if len(missing_pth3) > 0:
            print_dist(
                f"Missing {len(missing_pth3)} pth2's ({len(missing_pth3)/total_pth3 * 100:.1f}%), saving them to missing_pth3-{split}.txt"
            )
            # write_txt(missing_pth3, f"missing_pth3-{split}.txt")

        # Remove missing paths
        self.df = self.df[self.df["path1"].notna()]
        self.df = self.df[self.df["path2"].notna()]
        self.df = self.df[self.df["path_gen_tar"].notna()]
        self.df.reset_index(drop=True, inplace=True)

        self.max_words = max_words

        self.df["int1"] = self.df["pth1"].apply(lambda x: id2int(x, sub="0"))
        self.df["int2"] = self.df["pth2"].apply(lambda x: id2int(x, sub="0"))
        self.pairid2ref = self.df["int1"].to_dict()
        
        assert (
            self.df["int1"].nunique() == self.df["pth1"].nunique()
        ), "int1 is not unique"
        assert (
            self.df["int2"].nunique() == self.df["pth2"].nunique()
        ), "int2 is not unique"
        # int2id is a dict with key: int1, value: pth1
        self.int2id = self.df.groupby("int1")["pth1"].apply(set).to_dict()
        self.int2id = {k: list(v)[0] for k, v in self.int2id.items()}
        
        self.pairid2tar = self.df["int2"].to_dict()

        assert vid_query_method in [
            "middle",
            "random",
            "sample",
        ], f"Invalid vid_query_method: {vid_query_method}, must be one of middle, random, or sample"
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames
        
        print(f"WebVidCoVR {split} dataset in {mode} mode initialized")

    def _load_frame_as_pil(self, video_path):
        """
        Load a frame from video as PIL image without any transformation
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return PIL.Image.new('RGB', (384, 384))
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                print(f"Warning: No frames in video {video_path}")
                return PIL.Image.new('RGB', (384, 384))
                
            if self.vid_query_method == 'middle':
                frame_idx = frame_count // 2
            elif self.vid_query_method == 'random':
                frame_idx = random.randint(0, frame_count - 1)
                frame_idx = 0
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            cap.release()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                return PIL.Image.new('RGB', (384, 384))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return PIL.Image.fromarray(frame)
                
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return PIL.Image.new('RGB', (384, 384))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        ann = self.df.iloc[index]

        if self.mode == 'relative':
            reference_pth = str(ann["path1"])
            reference_pil = self._load_frame_as_pil(reference_pth)
            
            reference_vid = self.transform(reference_pil)

            gen_tar_pth = str(ann["path_gen_tar"])
            gen_tar_pil = PIL.Image.open(gen_tar_pth).convert('RGB')
            gen_tar_vid = self.transform(gen_tar_pil)

            caption = pre_caption(ann["edit"], self.max_words)
            ref_caption = str(ann["txt1"])
            tar_caption = str(ann["txt2"])

            target_pth = str(ann["path2"])
            target_pil = self._load_frame_as_pil(target_pth)
            target_vid = self.transform(target_pil)


            if self.split in ['train', 'val']:
                return {
                    'reference_image': reference_vid,
                    'gen_tar_img': gen_tar_vid,
                    'reference_name': ann["pth1"],
                    'target_image': target_vid,
                    'target_name': ann["pth2"],
                    'relative_caption': caption,
                    'ref_caption': ref_caption,
                    'target_caption': tar_caption,
                    'query_id': index
                }
            else:  # test
                return {
                    'reference_image': reference_vid,
                    'gen_tar_img': gen_tar_vid,
                    'reference_name': ann["pth1"],
                    'target_image': target_vid,
                    'target_name': ann["pth2"],
                    'relative_caption': caption,
                    'ref_caption': ref_caption,
                    'target_caption': tar_caption,
                    'query_id': index
                }
        elif self.mode == 'classic':
            vid_path = str(ann["path2"])
            vid_pil = self._load_frame_as_pil(vid_path)
            vid = self.transform(vid_pil)
            return {'image': vid, 'image_name': ann["pth2"]}
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
