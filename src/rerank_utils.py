
import json
import cv2
import re
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
from PIL import Image
from tqdm import tqdm


def build_frames_provider_webvid(
    video_base_dir: Union[str, Path],
    video_extension: str = "mp4",
    max_frames: int = 8,
    temp_dir: Optional[Path] = None
):
    vid_dir = Path(video_base_dir)

    def _provider(name: str) -> List[str]:
    
        return []
    
    return _provider

@torch.no_grad()
def qwen2vl_text_rerank(
    cache_path,
    uncertainty,
    similarities: np.ndarray,
    device: torch.device,
    tar_captions: List[str],
    reference_names: List[str],
    frames_provider,
    instructions: List[str],
    strategy: str = 'boost_max',
    uncertainty_thresh: float = 0.8,
    topk: int = 100,
    u_topk: int = 1000,
    lam: float = 0.2,
    boost_value: float = 100.0
) -> np.ndarray:

    return similarities
