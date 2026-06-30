#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig


def _init_logging():
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )


def load_qwen_model(model_path, lora_path, weight_name, cache_dir, use_quantization=True):
    """
    Load Qwen-Image model
    
    Args:
        model_path: Path to model
        lora_path: Path to LoRA weights
        weight_name: LoRA weight filename
        cache_dir: Cache directory
        use_quantization: Whether to use 8bit quantization
    
    Returns:
        Loaded pipeline
    """
    logging.info("Loading Qwen-Image model...")
    
    if use_quantization:
        try:
            logging.info("Using 8bit quantization")
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={
                    "load_in_8bit": True,
                    "bnb_8bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )

            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                device_map="cuda",
            )
            logging.info("Successfully loaded 8bit quantized model")
            
        except Exception as e:
            logging.warning(f"8bit quantized loading failed: {e}")
            logging.info("Falling back to normal loading")
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            logging.info("Successfully loaded normal model")
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        logging.info("Successfully loaded normal model")

    logging.info("Loading LoRA weights...")
    pipe.load_lora_weights(
        lora_path,
        weight_name=weight_name,
        cache_dir=cache_dir,
    )
    logging.info("LoRA weights loaded")
    
    return pipe


def generate_image_from_prompt(pipe, prompt, output_path, args):
    """
    Generate a single image from prompt
    
    Args:
        pipe: Qwen-Image pipeline
        prompt: Input prompt
        output_path: Output image path
        args: CLI arguments
        
    Returns:
        True if success, False otherwise
    """
    try:
        logging.info(f"Generating image: {output_path}")
        logging.info(f"Prompt: {prompt}")
        
        generator = torch.manual_seed(args.seed) if args.seed >= 0 else None
        
        image = pipe(
            prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.cfg_scale,
            generator=generator,
        ).images[0]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        image.save(output_path)
        logging.info(f"Image saved to: {output_path}")
        
        del image
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to generate image {output_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch generate CIRR dataset images using Qwen-Image from CSV file")
    
    parser.add_argument(
        "--csv_file",
        type=str,
        default="test_tar.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--col_name",
        type=str,
        default="tar_desc",
        help="Column name containing prompts"
    )
    parser.add_argument(
        "--query_id_name",
        type=str,
        default="query_id",
        help="Column name containing query IDs"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to Qwen-Image model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="lightx2v/Qwen-Image-Lightning",
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default="Qwen-Image-Lightning-4steps-V2.0.safetensors",
        help="LoRA weight filename"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="LoRA cache directory"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Generated image width"
    )
    parser.add_argument(
        "--height", 
        type=int,
        default=720,
        help="Generated image height"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, -1 to use random"
    )
    
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start processing CSV from this row (excluding header)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Stop processing at this row (excluding header), None = until end"
    )
    
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=True,
        help="Use 8bit quantization"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip if output image already exists"
    )
    
    args = parser.parse_args()
    
    _init_logging()
    
    logging.info(f"Generation parameters: {args}")
    
    pipe = load_qwen_model(
        args.model_path,
        args.lora_path,
        args.weight_name,
        args.cache_dir,
        args.use_quantization
    )
    
    logging.info(f"Reading CSV file: {args.csv_file}")
    image_tasks = []
    
    print(f"args.query_id_name: {args.query_id_name} | args.col_name: {args.col_name}")

    with open(args.csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if not row.get(args.query_id_name) or not row.get(args.col_name):
                continue
            
            if idx < args.start_index:
                continue
            if args.end_index is not None and idx >= args.end_index:
                break
            
            image_name = row[args.query_id_name] + '.jpg'
            output_path = os.path.join(args.output_dir, image_name)
            
            if args.skip_existing and os.path.exists(output_path):
                logging.info(f"Image already exists, skipping: {output_path}")
                continue
            
            image_tasks.append({
                'query_id': row[args.query_id_name],
                'prompt': row[args.col_name],
                'output_path': output_path
            })
    
    logging.info(f"{len(image_tasks)} images to generate")
    
    success_count = 0
    fail_count = 0
    
    for idx, task in enumerate(image_tasks):
        logging.info(f"\nProgress: {idx + 1}/{len(image_tasks)}")
        logging.info(f"query_id: {task['query_id']}")
        
        if generate_image_from_prompt(
            pipe,
            task['prompt'],
            task['output_path'],
            args
        ):
            success_count += 1
        else:
            fail_count += 1
    
    logging.info("\nGeneration finished!")
    logging.info(f"Success: {success_count}")
    logging.info(f"Fail: {fail_count}")
    logging.info(f"Total: {len(image_tasks)}")


if __name__ == "__main__":
    main()
