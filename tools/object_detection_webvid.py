#!/usr/bin/env python3
import os
GPU_ID = "0" 
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from rerank_utils import build_frames_provider_webvid
from tqdm import tqdm


VIDEO_BASE_DIR = ""  
VIDEO_EXTENSION = "mp4" 

CSV_FILE = ""

OBJ_DET_OUTPUT_DIR = ""
OBJ_DET_OUTPUT_FILE = "obj_det_ref.csv"
COL_NAME = "pth1" 

# Qwen2.5-VL model configuration
QWEN_VL_MODEL_PATH = ""

MAX_FRAMES = 8 

TEMP_FRAMES_DIR = ""

# Generation configuration
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3 
TOP_P = 0.9

# Qwen2.5-VL prompt template
DETECTION_PROMPT = """Analyze these video frames and list all the objects you can see.

TASK: Identify and list the main objects/entities present in the video frames.

INSTRUCTIONS:
1. Look at all the provided frames
2. Identify the key objects, people, animals, and things
3. Focus on concrete, visible objects (not abstract concepts)
4. List unique object types (don't count duplicates)
5. Use simple, common English words for objects
6. Output format: "The video contains object1, object2, object3." (no counts, no "and")

EXAMPLES:
...

IMPORTANT RULES:
- Use singular or plural forms naturally (e.g., "tree" or "trees", but be consistent)
- Don't include counts (e.g., NOT "3 persons", just "person")
- Separate objects with commas (no "and" before the last item)
- Keep it concise - focus on the 5-10 most prominent objects
- End with a period

Now analyze the video frames and output ONLY the object list in the specified format:"""


def initialize_qwen25_vl():
    """Initialize Qwen2.5-VL-7B-Instruct model"""
    print("Initializing Qwen2.5-VL-7B-Instruct model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not detected, using CPU (slower speed)")
    
    try:
        use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            cache_dir=QWEN_VL_MODEL_PATH,
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            cache_dir=QWEN_VL_MODEL_PATH,
            torch_dtype=use_dtype,
            device_map='auto'
        )
        
        print("Qwen2.5-VL model initialization completed")
        return processor, model, device
    except Exception as e:
        print(f"Qwen2.5-VL model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def detect_objects_with_qwen(processor, model, device, frame_paths):
    """
    Perform object detection on a set of frames using Qwen2.5-VL
    
    Args:
        processor: Qwen2.5-VL processor
        model: Qwen2.5-VL model
        device: computing device
        frame_paths: list of frame file paths
        
    Returns:
        obj_det_str: natural language description of object detection results
    """
    if not frame_paths:
        return "No frames extracted"
    
    try:
        # Build multimodal message: include all frames
        content = []
        
        # Add prompt
        content.append({"type": "text", "text": DETECTION_PROMPT})
        
        # Add video frames (maximum 8 frames)
        for frame_path in frame_paths[:MAX_FRAMES]:
            content.append({"type": "image", "image": frame_path})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # Process input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True
            )
            
            # Decode only the newly generated part
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            result = output_text[0] if isinstance(output_text, list) else str(output_text)
            result = result.strip()
            
            # Clean output: ensure correct format
            if not result.startswith("The video contains"):
                # Try to find standard format
                if "The video contains" in result:
                    start_idx = result.find("The video contains")
                    result = result[start_idx:]
                    if "." in result:
                        result = result[:result.find(".")+1]
                else:
                    # If no standard format, add prefix
                    result = f"The video contains {result}"
                    if not result.endswith("."):
                        result += "."
            
            # Further cleaning: extract first sentence
            if "." in result:
                result = result[:result.find(".")+1]
            
            return result
            
    except Exception as e:
        print(f"Qwen2.5-VL detection failed: {e}")
        return f"Detection failed: {str(e)}"


def sanitize_text(text: str) -> str:
    """Clean text: remove line breaks and extra whitespace"""
    if not isinstance(text, str):
        text = str(text)
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())
    return cleaned


def process_test_csv():
    """Process videos in test.csv for object detection using Qwen2.5-VL"""
    # Check if file exists
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        print(f"CSV file does not exist: {csv_path}")
        return
    
    # Create output directory
    obj_det_output_path = Path(OBJ_DET_OUTPUT_DIR)
    obj_det_output_path.mkdir(parents=True, exist_ok=True)
    obj_det_csv = obj_det_output_path / OBJ_DET_OUTPUT_FILE

    # Read existing obj_det for deduplication
    existing_ids = set()
    if obj_det_csv.exists():
        try:
            existing_df = pd.read_csv(obj_det_csv)
            if "video_id" in existing_df.columns:
                existing_ids = set(existing_df["video_id"].astype(str).str.strip())
                print(f"{len(existing_ids)} obj_det records already exist, will be skipped automatically during processing")
        except Exception as e:
            print(f"Failed to read existing {OBJ_DET_OUTPUT_FILE}: {e}")
    
    # Read CSV data
    print(f"Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return
    
    print(f"Found {len(df)} records")

    # Initialize Qwen2.5-VL model
    processor, model, device = initialize_qwen25_vl()
    if model is None:
        print("Failed to load Qwen2.5-VL model, exiting")
        return
    
    # Build frame extractor
    print("Building video frame extractor...")
    temp_dir = Path(TEMP_FRAMES_DIR) if TEMP_FRAMES_DIR else None
    frames_provider = build_frames_provider_webvid(
        video_base_dir=VIDEO_BASE_DIR,
        video_extension=VIDEO_EXTENSION,
        max_frames=MAX_FRAMES,
        temp_dir=temp_dir
    )
    print("Frame extractor built successfully")
    
    # Statistical information
    success_count = 0
    total_count = 0

    col_name = COL_NAME

    # Process each row of data
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing progress", unit="video"):
        # Skip empty rows
        if col_name not in row or pd.isna(row[col_name]):
            print(f"Skipping row {idx+1}: missing {col_name}")
            continue

        total_count += 1
        video_id = str(row[col_name]).strip()

        # Skip if already generated
        if video_id in existing_ids:
            continue
        
        print(f"\nProcessing row {idx+1}: {video_id}")
        
        # Get video frames using frame extractor
        frame_paths = frames_provider(video_id)
        
        if not frame_paths:
            print(f"Failed to extract video frames: {video_id}")
            # Write even if failed to avoid duplicate processing
            row_df = pd.DataFrame([{"video_id": video_id, "obj_det": "Frame extraction failed"}])
            row_df.to_csv(obj_det_csv, mode='a', header=not obj_det_csv.exists(), index=False)
            existing_ids.add(video_id)
            continue
        
        print(f"Extracted {len(frame_paths)} frames")
        
        # Perform object detection using Qwen2.5-VL
        print("Performing object detection with Qwen2.5-VL...")
        try:
            obj_det = detect_objects_with_qwen(processor, model, device, frame_paths)
            obj_det = sanitize_text(obj_det)
            print(f"Detection result: {obj_det}")
            
            # Append to CSV one by one
            row_df = pd.DataFrame([{"video_id": video_id, "obj_det": obj_det}])
            row_df.to_csv(obj_det_csv, mode='a', header=not obj_det_csv.exists(), index=False)
            existing_ids.add(video_id)
            success_count += 1
        except Exception as e:
            print(f"Object detection failed: {e}")
            # Write failure record
            row_df = pd.DataFrame([{"video_id": video_id, "obj_det": f"Detection failed: {str(e)}"}])
            row_df.to_csv(obj_det_csv, mode='a', header=not obj_det_csv.exists(), index=False)
            existing_ids.add(video_id)
            continue
    
    # Output statistical information
    print(f"\nProcessing completed!")
    print(f"Total processed: {total_count} videos")
    print(f"Successfully detected: {success_count} videos")
    print(f"Results saved to: {obj_det_csv}")


def main():
    """Main function"""
    print("Starting video object detection with Qwen2.5-VL...")
    print(f"Video directory: {VIDEO_BASE_DIR}")
    print(f"CSV file: {CSV_FILE}")
    print(f"Output directory: {OBJ_DET_OUTPUT_DIR}")
    print(f"Output file: {OBJ_DET_OUTPUT_FILE}")
    print(f"Maximum frames: {MAX_FRAMES}")
    print(f"Detection method: Qwen2.5-VL (VLM)")
    
    try:
        process_test_csv()
    except KeyboardInterrupt:
        print("\nUser interrupted operation")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()