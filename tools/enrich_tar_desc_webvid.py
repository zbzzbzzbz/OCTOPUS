#!/usr/bin/env python3
import os
GPU_ID = "0"  
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


ORIGINAL_CSV = ""
TARGET_DESC_CSV = ""
OBJ_DET_CSV = ""

OUTPUT_CSV = f""

QWEN_VL_MODEL_PATH = ""

PTH1_COL = "pth1"  
PTH2_COL = "pth2"  
DESC_COL = "tar_desc"  
EDIT_COL = "edit" 
EXIT_COL = "exit"  

MAX_NEW_TOKENS = 700  
TEMPERATURE = 0.3  
TOP_P = 0.9  

PROGRESS_INTERVAL = 5  

NUN_CAP = 10

ENRICH_PROMPT = """
You are a visual reasoning assistant that generates auxiliary titles for edited videos.

## GOAL
Given an edit instruction, detected objects from the reference video, and a generated target description,
produce **10 concise, realistic, and visually coherent auxiliary titles** that:
- Accurately reflect the edit instruction.
- Remain fully consistent with the detected objects and target description.
- Avoid introducing any new or imaginary objects, attributes, or settings.
- Each title should sound natural, descriptive, and contextually grounded.

## BEHAVIOR PRIORITY
1. **Faithfulness first** — every title must align with both the edit instruction and the detected objects.
2. **No hallucination** — do not invent unseen or unrelated content.
3. **Controlled variation** — each title should differ slightly in tone, focus, or phrasing, while keeping the same meaning.
4. **Concise expression** — 5–12 words per title, written in fluent English.
5. **Consistent formatting** — return titles as a valid JSON array of 10 strings, with no numbering or explanations.

## INPUT
Edit instruction: "{edit_instruction}"
Detected objects: {obj_det}
Generated target description: "{target_desc}"

## OUTPUT
Return exactly 10 auxiliary titles as a JSON array of strings.
Do not include any extra commentary or metadata.

### Example

Edit instruction: "change the person's shirt from white to red"
Detected objects: "person, shirt, park"
Generated target description: "A person wearing a red shirt walks in the park."
Output:
[
  "A person wearing a red shirt in the park",
  "A red-shirted person walking outdoors",
  "Casual walk through the park in red attire",
  "Man in red shirt strolling through greenery",
  "A person enjoying the park in a red top",
  "Bright red shirt contrasts with green park scenery",
  "A relaxed walk wearing a red shirt",
  "Park walk with person dressed in red",
  "A vivid red shirt seen in the park",
  "A person walks under trees wearing a red shirt"
]
"""


def sanitize_text(text: str) -> str:
    """Clean text: remove newlines and extra whitespace"""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


def initialize_qwen25_vl() -> Tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration, str]:
    """Initialize Qwen2.5-VL-7B-Instruct model"""
    print("Initializing Qwen2.5-VL-7B-Instruct model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("CUDA not detected, using CPU (slower speed)")
    
    try:
        use_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # First attempt to load from local path
        # Fallback to using HuggingFace ID and cache_dir
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
        
        print("Qwen2.5-VL model initialization completed successfully")
        return processor, model, device
        
    except Exception as e:
        print(f"Qwen2.5-VL model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def process_other_captions(output_text: list) -> list:
    """
    Process the auxiliary titles JSON array from model output
    
    Args:
        output_text: List of text from model output, expected to contain JSON-formatted auxiliary titles array
    
    Returns:
        list: Processed list of 10 titles (returns single-element list with original text if parsing fails)
    """
    import json
    import re
    
    if not output_text or len(output_text) == 0:
        return []
    
    # Get raw text from model output
    raw_text = output_text[0] if isinstance(output_text, list) else str(output_text)
    raw_text = sanitize_text(raw_text)
    
    try:
        # Try direct JSON parsing
        if raw_text.startswith('[') and raw_text.endswith(']'):
            captions_array = json.loads(raw_text)
            if isinstance(captions_array, list) and len(captions_array) > 0:
                # Return list of all titles
                cleaned_captions = [sanitize_text(str(caption)) for caption in captions_array]
                return cleaned_captions
        
        # If not direct JSON, try extracting JSON part from text
        json_match = re.search(r'\[(.*?)\]', raw_text, re.DOTALL)
        if json_match:
            json_str = '[' + json_match.group(1) + ']'
            captions_array = json.loads(json_str)
            if isinstance(captions_array, list) and len(captions_array) > 0:
                # Return list of all titles
                cleaned_captions = [sanitize_text(str(caption)) for caption in captions_array]
                return cleaned_captions
        
        # If JSON parsing fails, try extracting all titles from quotes
        quotes_matches = re.findall(r'"([^"]*)"', raw_text)
        if quotes_matches:
            # Clean and return all found titles
            cleaned_captions = [sanitize_text(caption) for caption in quotes_matches if caption.strip()]
            if cleaned_captions:
                return cleaned_captions
        
        # Try extracting multiple titles by line (handle non-JSON formatted lists)
        lines = raw_text.split('\n')
        extracted_captions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('[', ']', '{', '}')):
                # Remove possible sequence prefixes (e.g., "1. " or "- ")
                cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                cleaned_line = re.sub(r'^[-*]\s*', '', line)
                cleaned_line = re.sub(r'^["\']|["\']$', '', cleaned_line)  # Remove surrounding quotes
                if cleaned_line:
                    extracted_captions.append(sanitize_text(cleaned_line))
        
        if extracted_captions:
            return extracted_captions[:NUN_CAP]  # Take up to NUN_CAP titles
        
        # If all parsing methods fail, return single-element list with original text
        return [raw_text]
        
    except (json.JSONDecodeError, IndexError, TypeError) as e:
        print(f"JSON parsing failed, attempting text extraction: {e}")
        
        # Try simple text extraction: find non-empty multi-line content
        lines = raw_text.split('\n')
        extracted_captions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('[', ']', '{', '}')):
                # Remove possible sequence prefixes (e.g., "1. " or "- ")
                cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                cleaned_line = re.sub(r'^[-*]\s*', '', line)
                cleaned_line = re.sub(r'^["\']|["\']$', '', cleaned_line)  # Remove surrounding quotes
                if cleaned_line:
                    extracted_captions.append(sanitize_text(cleaned_line))
        
        if extracted_captions:
            return extracted_captions[:NUN_CAP]  # Take up to NUN_CAP titles
        
        return [raw_text]

def enrich_description(processor: AutoProcessor,
                      model: Qwen2_5_VLForConditionalGeneration,
                      device: str,
                      edit_instruction: str,
                      target_desc: str,
                      obj_det: str) -> str:
    """Evaluate and enrich target description based on edit instruction and object detection results"""
    
    # Clean inputs
    edit_clean = sanitize_text(edit_instruction)
    target_clean = sanitize_text(target_desc)
    obj_det_clean = sanitize_text(obj_det)
    
    # Truncate to first 20 words
    obj_det_words = obj_det_clean.split()
    if len(obj_det_words) > 20:
        obj_det_clean = " ".join(obj_det_words[:20]) + "..."
    else:
        obj_det_clean = " ".join(obj_det_words)
    
    # Construct prompt
    prompt_text = ENRICH_PROMPT.format(
        edit_instruction=edit_clean,
        target_desc=target_clean,
        obj_det=obj_det_clean
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text}
        ]
    }]
    
    try:
        # Process input (text only, no images needed)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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

            captions_list = process_other_captions(output_text)
            if len(captions_list) != NUN_CAP:
                print(f"Warning: len(captions_list) {len(captions_list)} != {NUN_CAP} ")
            import json
            result = json.dumps(captions_list, ensure_ascii=False)
  
            return result
            
    except Exception as e:
        print(f"Description enrichment failed: {e}")
        return target_desc  # Return original description on failure


def main():
    """Main function"""
    print("Starting target description enrichment...")
    print(f"Original CSV: {ORIGINAL_CSV}")
    print(f"Target Description CSV: {TARGET_DESC_CSV}")
    print(f"Object Detection CSV: {OBJ_DET_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Model Path: {QWEN_VL_MODEL_PATH}")
    print(f"Generation Configuration: max_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"\nNote: Evaluate and enrich target video (pth2) descriptions using edit instructions (edit/exit) and object detection results from reference videos (pth1)")
    
    # Check input files
    original_path = Path(ORIGINAL_CSV)
    target_desc_path = Path(TARGET_DESC_CSV)
    obj_det_path = Path(OBJ_DET_CSV)
    
    if not original_path.exists():
        print(f"Original CSV does not exist: {original_path}")
        sys.exit(1)
    
    if not target_desc_path.exists():
        print(f"Target Description CSV does not exist: {target_desc_path}")
        sys.exit(1)
    
    if not obj_det_path.exists():
        print(f"Object Detection CSV does not exist: {obj_det_path}")
        sys.exit(1)
    
    output_path = Path(OUTPUT_CSV)
    
    # Read original CSV to build pth2 -> pth1 mapping
    print(f"\nReading original CSV: {original_path}")
    try:
        original_df = pd.read_csv(original_path)
    except Exception as e:
        print(f"Failed to read original CSV: {e}")
        sys.exit(1)
    
    if PTH1_COL not in original_df.columns or PTH2_COL not in original_df.columns:
        print(f"Original CSV must contain columns {PTH1_COL} and {PTH2_COL}")
        sys.exit(1)
    
    # Determine which edit column to use
    edit_col_to_use = EDIT_COL if EDIT_COL in original_df.columns else (EXIT_COL if EXIT_COL in original_df.columns else None)
    if edit_col_to_use is None:
        print(f"Original CSV missing edit instruction column: neither '{EDIT_COL}' nor '{EXIT_COL}' exists")
        sys.exit(1)
    
    print(f"Using edit column: {edit_col_to_use}")
    
    # Build pth2 -> (pth1, edit) mapping dictionary
    print("Building pth2 -> (pth1, edit) mapping...")
    pth2_to_info = {}
    for _, row in original_df.iterrows():
        pth1 = str(row[PTH1_COL]).strip()
        pth2 = str(row[PTH2_COL]).strip()
        edit = str(row[edit_col_to_use]).strip() if pd.notna(row[edit_col_to_use]) else ""
        pth2_to_info[pth2] = {"pth1": pth1, "edit": edit}
    
    print(f"Successfully built {len(pth2_to_info)} pth2->(pth1, edit) mappings")
    
    # Read CSV files
    print(f"Reading Target Description CSV: {target_desc_path}")
    try:
        target_df = pd.read_csv(target_desc_path)
    except Exception as e:
        print(f"Failed to read Target Description CSV: {e}")
        sys.exit(1)
    
    print(f"Reading Object Detection CSV: {obj_det_path}")
    try:
        obj_det_df = pd.read_csv(obj_det_path)
    except Exception as e:
        print(f"Failed to read Object Detection CSV: {e}")
        sys.exit(1)
    
    # Column checks
    # Target_desc CSV should contain pth2 and tar_desc columns
    # For compatibility, check if "target" column exists (i.e., pth2)
    if "target" not in target_df.columns and PTH2_COL not in target_df.columns:
        print(f"Target Description CSV missing column: target or {PTH2_COL}")
        sys.exit(1)
    
    # Determine actual pth2 column name used
    pth2_col_in_desc = "target" if "target" in target_df.columns else PTH2_COL
    
    if DESC_COL not in target_df.columns:
        print(f"Target Description CSV missing column: {DESC_COL}")
        sys.exit(1)
    
    if "video_id" not in obj_det_df.columns or "obj_det" not in obj_det_df.columns:
        print("Object Detection CSV must contain columns video_id and obj_det")
        sys.exit(1)
    
    # Create object detection dictionary (pth1 -> obj_det)
    print("Building object detection mapping (pth1 -> obj_det)...")
    obj_det_dict = {}
    for _, row in obj_det_df.iterrows():
        pth1 = str(row["video_id"]).strip()
        obj_det = str(row["obj_det"]).strip()
        obj_det_dict[pth1] = obj_det
    
    print(f"Loaded {len(obj_det_dict)} object detection records (pth1)")
    
    # Initialize model
    processor, model, device = initialize_qwen25_vl()
    if model is None:
        print("Failed to load Qwen2.5-VL model, exiting")
        sys.exit(1)
    
    # Process row by row
    print(f"\nStarting target description enrichment...")
    enriched_desc_list = []
    total = len(target_df)
    
    matched_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(target_df.iterrows(), total=total, desc="Enrichment Progress", unit="row"):
        pth2 = str(row[pth2_col_in_desc]).strip()  # Target video ID
        target_desc = row.get(DESC_COL, "")
        
        # Handle empty descriptions
        if pd.isna(target_desc) or str(target_desc).strip() == "":
            enriched_desc_list.append("")
            skipped_count += 1
            continue
        
        target_desc_str = str(target_desc).strip()
        
        # Get corresponding pth1 and edit via pth2
        info = pth2_to_info.get(pth2, None)
        
        if info is None:
            # No matching information found, keep original description
            enriched_desc_list.append(target_desc_str)
            skipped_count += 1
            continue
        
        pth1 = info["pth1"]
        edit_instruction = info["edit"]
        
        # If no edit instruction, keep original description
        if not edit_instruction or edit_instruction == "":
            enriched_desc_list.append(target_desc_str)
            skipped_count += 1
            continue
        
        # Find corresponding object detection results (using pth1)
        obj_det = obj_det_dict.get(pth1, None)
        
        if obj_det is None or obj_det == "" or "failed" in obj_det.lower():
            # No object detection results, keep original description
            enriched_desc_list.append(target_desc_str)
            skipped_count += 1
            continue
        
        # Use model to evaluate and enrich description
        try:
            enriched_desc = enrich_description(
                processor, 
                model, 
                device,
                edit_instruction,
                target_desc_str,
                obj_det
            )
            enriched_desc_list.append(enriched_desc)
            matched_count += 1
            
            # Regularly output progress
            if (idx + 1) % PROGRESS_INTERVAL == 0:
                print(f"\n{'='*80}")
                print(f"Target Video (pth2): {pth2}")
                print(f"Reference Video (pth1): {pth1}")
                print(f"Edit Instruction: {edit_instruction}")
                print(f"Detection Results: {obj_det}")
                print(f"Original Description: {target_desc_str}")
                print(f"Enriched Description: {enriched_desc}")
                print(f"Progress: {idx+1}/{total}")
                
        except Exception as e:
            print(f"\nWarning: Failed to process row {idx+1}/{total}: {e}")
            enriched_desc_list.append(target_desc_str)
            skipped_count += 1
    
    # Create output DataFrame
    result_df = target_df.copy()
    result_df["enriched_desc"] = enriched_desc_list
    
    # Write to CSV
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"Processing completed!")
        print(f"Total processed: {total} rows")
        print(f"Successfully enriched: {matched_count} rows")
        print(f"Skipped/kept original: {skipped_count} rows")
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to write to CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupted operation")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()