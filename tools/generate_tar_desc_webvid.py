#!/usr/bin/env python3
import os
import pandas as pd
import base64
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import requests
import cv2


OPENAI_API_KEY = ""  
OPENAI_API_BASE = ""  
OPENAI_MODEL = "gpt-4o-mini"  

VIDEO_BASE_DIR = "" 
VIDEO_EXTENSION = "mp4" 
CSV_FILE = ""

PROMPT_MODE = "cot"

OUTPUT_DIR = ""
OUTPUT_FILE = f"{OPENAI_MODEL}_vid_edit_{PROMPT_MODE}.csv"

REF_VIDEO_COL = "pth1"  
TARGET_VIDEO_COL = "pth2"  
EDIT_COL = "edit"  

MAX_FRAMES = 1

# Temporary frame directory (optional, if not provided, use temp_frames in the directory where the script is located)
TEMP_FRAMES_DIR = ""

MAX_TOKENS = 1000
TEMPERATURE = 0.3 

IMAGE_MAX_SIZE = 1024 
IMAGE_QUALITY = 85  


GEN_TAR_CAPTION_COT = '''
- You are an image description expert. You are given an original image and manipulation text.
- Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.
- You should carefully generate an image description of the target image with a thought of your understanding of the manipulation intents.

## Guidelines on generating the Original Image Description

    - Ensure that the original image description is thorough and detailed, capturing all visible objects, attributes, and elements. Specific attention should be given to any objects breeds, relationships, color, scenes, and the overarching domain of the image to provide a complete understanding.
    - The original image description should be as accurate as possible, reflecting the content and context of the image. 

## Guidelines on generating the Thoughts
    - In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target image description.
    - Provide insight into how you interpreted the manipulation intent detailed in the manipulation text, considering various semantic aspects.
    - Conclude with how these understandings were utilized to formulate the target image description, ensuring a logical and visually coherent transformation.

### Guidelines on generating the Reflections
    - In your Reflections, summarize how the manipulation intent influenced your approach to transforming the original image description.
    - Explain how the changes made reflect the specific semantic aspects involved, such as addition, negation, spatial relations, or viewpoint.
    - Highlight key decisions that were made to preserve the coherence and context of the original image while meeting the manipulation intent.
    - Reflect on the impact these changes have on the overall appearance or narrative of the image.
    - Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies applied in crafting the target image description, offering a logical connection between the original content and the final description.

## Guidelines on generating Target Image Description
    - The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:
    - The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible. Here are some more examples for reference:

## On the input format <Input>
- Input consist of two parts: The original image and the manipulation text.
{
    "Original Image": <image>,
    "Manipulation text": <manipulation_text>.
}
    - The original image is provided as an image input.
    - The manipulation text is the text that describes the changes to be made to the original image.

## Guidelines on determining the response <Response>
- Responses consist of Four parts: the Original Image Description, Thoughts, Reflections, and Target Image Description.
{
    "Original Image Description": <original_image_description>,
    "Thoughts": <thoughts>,
    "Reflections": <reflections>,
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Original Image": <image>,
    "Manipulation text": "Shows a dog of the same breed with darker brown fur laying in the grass with a woven basket of flowers."
}
<Response>
{
    "Original Image Description": "The image shows a man in a green field with a dog of Hungarian Vizsla breed. The man, wearing a dark green jacket and cap, is standing while the dog is pointing, indicating a hunting or tracking activity. The background is a grassy field extending to the horizon with trees faintly visible in the distance. The atmosphere appears calm, with an overcast sky providing soft lighting for the scene.",
    "Thoughts": "The manipulation intent is to change the scene by introducing a dog of Hungarian Vizsla breed but with darker brown fur, and to have it laying in the grass with a woven basket of flowers. This involves altering the dog's position and color, as well as adding a new element, the basket of flowers, to the scene.",
    "Reflections": "I reflect that the manipulation focuses on changing the dog's appearance and activity while adding a new object to the scene, which shifts the focus from the interaction between the person and the dog to a more serene and decorative setting.",
    "Target Image Description": "A Hungarian Vizsla dog with darker brown fur is laying in the grass next to a woven basket of flowers."
}

## Example 2
<Input>
{
    "Original Image": <image>,
    "Manipulation text": "Focus on a single bird with open wings and get closer"
}
<Response>
{
    "Original Image Description": "The image shows a group of black vultures perched on a metal structure. The birds are gathered closely together, with two of them touching beaks, forming a heart shape. Several birds have their wings folded, while one is displaying open wings. The background features a blurred landscape, suggesting an elevated or open area.",
    "Thoughts": "The manipulation intent is to focus on a single bird with open wings and to get closer, which suggests a change in both the subject focus and the perspective. The original image features a group of vultures, but the manipulation requires isolating one bird, particularly one with open wings, and zooming in to provide a closer view. This would emphasize the details of the bird's wings and features, shifting the focus from the group to an individual bird.",
    "Reflections": "I reflect that the manipulation intent involves changing the focus from a group to a single bird, enhancing the details and features of the bird by getting closer, which alters the narrative from a collective scene to an individual focus.",
    "Target Image Description": "A single black vulture with open wings is shown up close."
}

## Example 3
<Input>
{   
    "Original Image": <image>,
    "Manipulation text": "Replace entire content with saloon spot with man and boy."
}
<Response>
{
    "Original Image Description": "The image features two miniature Schnauzer dogs standing on a mat near a glass door. One dog has a red collar, and they are positioned close to each other, facing in the same direction, with one appearing to sniff the other. The background includes a folding chair visible outside through the glass door, a curtain with decorative patterns, and part of an indoor cabinet with household items.",
    "Thoughts": "The manipulation intent was to replace the original content featuring two dogs with an entirely different setting involving a barbershop where a man and a young boy are present. This includes changing the focus from pets to human characters, transforming the context into a typical barbershop scene. The new background features elements such as barber tools, a window with blinds, and a more human-oriented environment that provides a sense of familiarity and warmth. This manipulation significantly shifts the original focus and dynamics, changing both the subjects and the setting entirely to create a human-centered narrative.",
    "Reflections": "I reflect that the manipulation intent involved creating an entirely different scene by introducing human subjects and a barbershop environment. This required me to focus on capturing the new setting details, including the interaction between the man and the boy, the objects they interact with, and the new atmosphere that evokes a sense of everyday life.",
    "Target Image Description": "A man giving a young boy a haircut in a barbershop."
}

'''



def extract_middle_frame(video_path: Path, output_path: Path = None) -> str:
    if not video_path.exists():
        print(f"The video file does not exist: {video_path}")
        return ""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return ""
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"The video file is damaged or empty: {video_path}")
        return ""
    
    middle_frame_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"Unable to read intermediate frames: {video_path}")
        return ""
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    if output_path is None:
        if TEMP_FRAMES_DIR:
            temp_dir = Path(TEMP_FRAMES_DIR)
        else:
            temp_dir = Path(__file__).parent / "temp_frames"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"{video_path.stem}_middle.jpg"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pil_image.save(output_path, quality=95)
    
    return str(output_path)


def encode_image_to_base64(image_path: Path, max_size: int = 1024, quality: int = 85) -> str:
    try:
        img = Image.open(image_path)
        
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    except Exception as e:
        print(f"Image encoding failed: {e}")
        return ""


def check_api_key():
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY !!")
        return False
    
    print(f"API configuration completed (base={OPENAI_API_BASE}, model={OPENAI_MODEL})")
    return True

def parse_cot_output(cot_output: str) -> dict:
    import re
    import json
    
    result = {
        "original_image_description": "",
        "thoughts": "",
        "reflections": "",
        "target_image_description": ""
    }
    
    if not isinstance(cot_output, str):
        cot_output = str(cot_output)
    
    text = cot_output.strip()
    

    try:
        json_match = re.search(r'\{[\s\S]*"Target Image Description"[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            result["original_image_description"] = parsed.get("Original Image Description", "")
            result["thoughts"] = parsed.get("Thoughts", "")
            result["reflections"] = parsed.get("Reflections", "")
            result["target_image_description"] = parsed.get("Target Image Description", "")
            return result
    except:
        pass
    
    patterns = {
        "original_image_description": r'"Original Image Description"\s*[:：]\s*"([^"]+)"',
        "thoughts": r'"Thoughts"\s*[:：]\s*"([^"]+)"',
        "reflections": r'"Reflections"\s*[:：]\s*"([^"]+)"',
        "target_image_description": r'"Target Image Description"\s*[:：]\s*"([^"]+)"'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()
    
    if not result["target_image_description"]:
        m = re.search(r"(?i)(?:final[_\s-]?caption|target[_\s-]?image[_\s-]?description)\s*[:：]\s*(.+)", text)
        if m:
            result["target_image_description"] = m.group(1).strip()
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                result["target_image_description"] = lines[-1]
    
    return result

def generate_target_description(frame_paths, edit_instruction, prompt_mode="cot"):
    if not frame_paths:
        return "No frames extracted"
    
    try:
        content = []
        
        for frame_path in frame_paths[:MAX_FRAMES]:
            base64_image = encode_image_to_base64(frame_path, IMAGE_MAX_SIZE, IMAGE_QUALITY)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
        
        if prompt_mode.lower() == "cot":
            prompt_text = GEN_TAR_CAPTION_COT + f"""

            Now, please analyze the following:

            <Input>
            {{
                "Original Image": <image>,
                "Manipulation text": "{str(edit_instruction)}"
            }}

            Please provide your response in the same format as shown in the examples above.
            """
        else:
            raise ValueError(f"Invalid prompt_mode: {prompt_mode}")
        
        content.append({
            "type": "text",
            "text": prompt_text
        })
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        
        response_data = response.json()
        result = response_data['choices'][0]['message']['content'].strip()
        
        if "cot" in prompt_mode.lower():
            parsed = parse_cot_output(result)
            
            print("\n" + "="*80)
            print("COT:")
            print("="*80)
            
            if parsed["original_image_description"]:
                print(f"\nOriginal Image Description:")
                print(f"   {parsed['original_image_description']}")
            
            if parsed["thoughts"]:
                print(f"\nThoughts:")
                print(f"   {parsed['thoughts']}")
            
            if parsed["reflections"]:
                print(f"\nReflections:")
                print(f"   {parsed['reflections']}")
            
            if parsed["target_image_description"]:
                print(f"\nTarget Image Description:")
                print(f"   {parsed['target_image_description']}")
            
            print("="*80 + "\n")
            
            result = parsed["target_image_description"] if parsed["target_image_description"] else result
        
        return result
            
    except Exception as e:
        print(f"GPT-4o-mini Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Generation failed: {str(e)}"


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())
    return cleaned


def process_test_csv():
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        print(f"No CSV file found at: {csv_path}")
        return

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_path / OUTPUT_FILE

    existing_data = {}
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            if "target" in existing_df.columns and "tar_desc" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    target = str(row["target"]).strip()
                    tar_desc = str(row["tar_desc"]).strip() if pd.notna(row["tar_desc"]) else ""
                    existing_data[target] = tar_desc
                print(f"Found {len(existing_data)} existing records; skipping based on tar_desc quality during processing.")
        except Exception as e:
            print(f"Failed to read existing output file {OUTPUT_FILE}: {e}")

    print(f"CSV path: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Found {len(df)} records in the input CSV")

    # Validate required columns
    if REF_VIDEO_COL not in df.columns:
        print(f"Missing reference video column: {REF_VIDEO_COL}")
        return
    if TARGET_VIDEO_COL not in df.columns:
        print(f"Missing target video column: {TARGET_VIDEO_COL}")
        return
    if EDIT_COL not in df.columns:
        print(f"Missing edit instruction column: {EDIT_COL}")
        return

    print(f"Using edit instruction column: {EDIT_COL}")

    # Check API key
    if not check_api_key():
        print("API configuration check failed. Exiting.")
        return

    print("Preparing to extract middle frames from videos...")

    success_count = 0
    total_count = 0

    # Iterate over each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="sample"):
        # Skip rows with missing refs or targets
        if pd.isna(row.get(REF_VIDEO_COL)) or pd.isna(row.get(TARGET_VIDEO_COL)):
            print(f"Skipping row {idx+1}: missing reference or target video ID")
            continue

        total_count += 1
        ref_video_id = str(row[REF_VIDEO_COL]).strip()
        target_video_id = str(row[TARGET_VIDEO_COL]).strip()
        edit_instruction = str(row.get(EDIT_COL, "")).strip() if not pd.isna(row.get(EDIT_COL)) else ""

        # If already generated and tar_desc appears valid (exists and length>=2), skip
        if target_video_id in existing_data:
            existing_tar_desc = existing_data[target_video_id]
            if existing_tar_desc and len(existing_tar_desc) >= 2:
                continue
            else:
                print(f"Target {target_video_id} exists but tar_desc is invalid (missing or too short); regenerating.")

        print(f"\nProcessing row {idx+1}: ref={ref_video_id}, target={target_video_id}")

        # Construct reference video path
        video_path = Path(VIDEO_BASE_DIR) / f"{ref_video_id}.{VIDEO_EXTENSION}"

        # Extract middle frame from reference video
        frame_path = extract_middle_frame(video_path)

        if not frame_path:
            print(f"Failed to extract middle frame for reference video: {ref_video_id}")
            # Write a record even on failure to avoid repeated attempts
            row_df = pd.DataFrame([{"target": target_video_id, "tar_desc": "Frame extraction failed"}])
            row_df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
            existing_data[target_video_id] = "Frame extraction failed"
            continue

        print(f"Extracted middle frame: {frame_path}")
        print(f"Edit instruction: {edit_instruction}")

        frame_paths = [frame_path]

        print(f"Generating target image description... (model={OPENAI_MODEL}, prompt_mode={PROMPT_MODE})")
        try:
            tar_desc = generate_target_description(frame_paths, edit_instruction, prompt_mode=PROMPT_MODE)
            tar_desc = sanitize_text(tar_desc)
            print(f"Generated result: {tar_desc}")

            # Append single-row result to CSV
            row_df = pd.DataFrame([{"target": target_video_id, "tar_desc": tar_desc}])
            row_df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
            existing_data[target_video_id] = tar_desc
            success_count += 1
        except Exception as e:
            print(f"Target description generation failed: {e}")
            error_msg = f"Generation failed: {str(e)}"
            row_df = pd.DataFrame([{"target": target_video_id, "tar_desc": error_msg}])
            row_df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
            existing_data[target_video_id] = error_msg
            continue

    # Summary
    print("\nProcessing complete.")
    print(f"Total samples processed: {total_count}")
    print(f"Successfully generated: {success_count}")
    print(f"Results saved to: {output_csv}")


def main():
    """
    Main entry point.
    """
    print("Starting generation of target image descriptions from reference video middle frames.")
    print(f"Video base directory: {VIDEO_BASE_DIR}")
    print(f"Input CSV file: {CSV_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("Frame extraction method: extract middle frame from video (no external API required)")
    print(f"Generation method: {OPENAI_MODEL} (reference frame + edit instruction)")
    print(f"Prompt mode: {PROMPT_MODE}")
    print(f"Model: {OPENAI_MODEL}")

    try:
        process_test_csv()
    except KeyboardInterrupt:
        print("\nUser interrupted the process.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()