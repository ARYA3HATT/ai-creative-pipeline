import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def get_subtitle(frame_idx, fps):
    time_sec = frame_idx / fps
    
    if 0 <= time_sec < 54:
        return "Initializing the AI Product Creative Generation Pipeline. LangGraph orchestrates 7 sequential agent nodes powered by Groq and Llama."
    elif 54 <= time_sec < 162:
        return "5 marketing images generated via Pollinations.ai FLUX — zero API key, zero cost. 2 video reels via video pipeline. Production routes through ComfyUI + Wan 2.1 on GPU."
    elif 162 <= time_sec < 270:
        return "Full execution audit trail — every agent output is inspectable and logged to SQLite in real time."
    elif 270 <= time_sec < 351:
        return "Critic node runs 7 parallel VLM evaluations. Score below 7.0 triggers automatic prompt mutation and re-generation via LangGraph conditional edge."
    elif 351 <= time_sec < 459:
        return "Bulk processing via Celery + Redis. Each URL runs in an isolated Celery task — individual failures never halt the batch."
    elif 459 <= time_sec <= 540:
        return "Core engineering: LangGraph conditional retry edges, ComfyUI process isolation, Celery per-row fault isolation for enterprise-scale bulk processing."
    return ""

def wrap_text(text, font, max_width):
    lines = []
    words = text.split(' ')
    current_line = []
    
    for word in words:
        current_line.append(word)
        # Check size of line
        line_str = ' '.join(current_line)
        bbox = font.getbbox(line_str)
        width = bbox[2] - bbox[0]
        if width > max_width:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(' '.join(current_line))
        
    return lines

def main():
    input_video = "scratch/raw_recording.mp4"
    output_video = "demo_video.mp4"
    
    print(f"Opening input video: {input_video}")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return
        
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25.0
        
    print(f"Properties: {width}x{height} @ {fps} fps")
    
    # Try different codecs: mp4v is highly compatible
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Try to load a premium sans-serif font
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Menlo.ttc"
    ]
    font = None
    for p in font_paths:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, 36)
                print(f"Loaded font: {p}")
                break
            except Exception:
                pass
    if font is None:
        # Fallback to default
        font = ImageFont.load_default()
        print("Using default PIL font")
        
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        subtitle_text = get_subtitle(frame_idx, fps)
        
        if subtitle_text:
            # Convert OpenCV image (BGR) to PIL image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            # 1. Draw a clean, modern semi-transparent subtitle background bar
            # Bar coordinates: bottom section of the viewport
            bar_height = 120
            bar_top = height - bar_height - 30
            bar_bottom = height - 30
            
            # Draw semi-transparent rectangle
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([100, bar_top, width - 100, bar_bottom], fill=(15, 23, 42, 210)) # Slate-900 with premium opacity
            
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(pil_img)
            
            # 2. Text layout and alignment
            lines = wrap_text(subtitle_text, font, width - 260)
            
            # Center the lines vertically in the bar
            total_text_height = 0
            line_heights = []
            for line in lines:
                bbox = font.getbbox(line)
                h = bbox[3] - bbox[1]
                line_heights.append(h)
                total_text_height += h + 6
                
            y = bar_top + (bar_height - total_text_height) // 2
            
            for line, h in zip(lines, line_heights):
                bbox = font.getbbox(line)
                w = bbox[2] - bbox[0]
                x = (width - w) // 2
                draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
                y += h + 6
                
            # Convert back to BGR numpy array
            frame_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        else:
            out.write(frame)
            
        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({int(frame_idx/total_frames*100)}%)")
            
    cap.release()
    out.release()
    print("Compilation finished successfully! demo_video.mp4 created.")

if __name__ == "__main__":
    main()
